"""
FastAPI Application.

Module: M12 (FastAPI Layer)

Endpoints:
    POST /analyze              -> upload dataset(s) + query, run the full pipeline synchronously
    GET  /trace/{session_id}   -> execution trace timeline for a session
    GET  /report/{session_id}  -> final markdown report for a session
    GET  /chart/{session_id}/{filename} -> serve a generated chart image
    GET  /health                -> basic liveness check

Execution is synchronous within the request (see PRD Section 6: no task
queue). Fine for small datasets and a single-user demo context.
"""
import io
import os
import json
import math
import traceback

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.db.connection import init_db, get_connection
from app.core.session_runner import run_session
from app.core.tracing import get_trace_for_session

app = FastAPI(title="Adaptive Agent Orchestration Framework", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a local student-project demo
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(
    files: list[UploadFile] = File(...),
    query: str = Form(...),
):
    """
    Accepts one or more CSV files plus a natural-language analysis query.
    Runs the full adaptive pipeline synchronously and returns the final state
    (minus raw dataframes, which aren't JSON-serializable) plus the session_id.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    dataframes = []
    dataset_names = []

    for f in files:
        if not f.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"Only CSV files are supported: {f.filename}")
        content = f.file.read()
        if not content or len(content.strip()) == 0:
            raise HTTPException(status_code=400, detail=f"File is empty: {f.filename}")
        try:
            df = pd.read_csv(io.BytesIO(content))
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail=f"File has no parsable data: {f.filename}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse {f.filename}: {e}")
        if df.shape[0] == 0:
            raise HTTPException(
                status_code=400,
                detail=f"'{f.filename}' has headers but no data rows.",
            )
        if df.shape[1] == 0:
            raise HTTPException(status_code=400, detail=f"'{f.filename}' has no columns.")
        dataframes.append(df)
        dataset_names.append(f.filename)

    if not dataframes:
        raise HTTPException(status_code=400, detail="No valid files uploaded.")

    try:
        result = run_session(dataframes, dataset_names, query)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(exc)}")

    return _serialize_result(result)


@app.get("/trace/{session_id}")
def get_trace(session_id: str):
    trace = get_trace_for_session(session_id)
    if not trace:
        raise HTTPException(status_code=404, detail="No trace found for this session.")
    return {"session_id": session_id, "trace": trace}


@app.get("/report/{session_id}")
def get_report(session_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT session_id, user_query, complexity_level, complexity_score, routing_reason "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found.")
    return dict(row)


@app.get("/report/{session_id}/download")
def download_report_pdf(session_id: str):
    """
    Serves the complete PDF report (all sections + embedded chart images)
    generated at the end of the pipeline run for this session.
    """
    pdf_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "outputs", session_id, "report.pdf"
    )
    pdf_path = os.path.normpath(pdf_path)
    if not os.path.isfile(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="PDF report not found for this session. It may not have finished generating.",
        )
    return FileResponse(
        pdf_path, media_type="application/pdf", filename=f"analysis_report_{session_id[:8]}.pdf"
    )


@app.get("/dataset/{session_id}/download-cleaned")
def download_cleaned_dataset(session_id: str):
    """
    Serves the cleaned CSV dataset generated during the pipeline run for this session.
    """
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "outputs", session_id, "cleaned_dataset.csv"
    )
    csv_path = os.path.normpath(csv_path)
    if not os.path.isfile(csv_path):
        raise HTTPException(
            status_code=404,
            detail="Cleaned dataset not found for this session.",
        )
    return FileResponse(
        csv_path, media_type="text/csv", filename=f"cleaned_dataset_{session_id[:8]}.csv"
    )


@app.get("/chart/{session_id}/{filename}")
def get_chart(session_id: str, filename: str):
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "outputs", session_id, filename
    )
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(path)


def _clean_for_json(obj):
    """Recursively replace NaN, Infinity, -Infinity with None for RFC-compliant JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(x) for x in obj]
    return obj


def _serialize_result(state: dict) -> dict:
    """Strip non-JSON-serializable fields (dataframes) and sanitize NaNs before returning."""
    exclude = {"raw_datasets", "cleaned_data"}
    serialized = {k: v for k, v in state.items() if k not in exclude}

    cleaned = state.get("cleaned_data")
    if cleaned and len(cleaned) > 0 and isinstance(cleaned[0], pd.DataFrame):
        df_clean = cleaned[0]
        serialized["cleaned_csv_available"] = True
        serialized["cleaned_csv_filename"] = state.get("cleaned_csv_filename", "cleaned_dataset.csv")
        serialized["cleaned_rows"] = int(len(df_clean))
        serialized["cleaned_columns"] = [str(c) for c in df_clean.columns]
        # Clean preview records to avoid NaNs
        preview_df = df_clean.head(25).fillna("")
        serialized["cleaned_preview"] = preview_df.to_dict(orient="records")
        # Only embed inline CSV text for smaller files to avoid bloating HTTP payload
        if len(df_clean) <= 500:
            serialized["cleaned_csv_text"] = df_clean.to_csv(index=False)
        else:
            serialized["cleaned_csv_text"] = None
    else:
        serialized["cleaned_csv_available"] = False
        serialized["cleaned_csv_text"] = None

    return _clean_for_json(serialized)
