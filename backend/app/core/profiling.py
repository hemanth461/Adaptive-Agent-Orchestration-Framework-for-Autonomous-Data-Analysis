"""
Dataset Profiling Module.

Module: M1 (Data Ingestion & Profiling)

Deterministic, pandas-only profiling -- no LLM calls here. Produces the
`dataset_profile` dict that feeds both the Complexity Scoring Engine (M3)
and the Data Preparation Agent (M6).
"""
import pandas as pd
import numpy as np


def profile_single_dataset(df: pd.DataFrame, name: str) -> dict:
    """Profile one dataframe: shape, dtypes, missing values, duplicates."""
    n_rows, n_cols = df.shape

    if n_rows == 0 or n_cols == 0:
        # Edge case: empty upload (0 rows or 0 columns). Return a well-formed
        # but empty profile rather than letting pandas raise on mean-of-empty.
        return {
            "name": name,
            "n_rows": n_rows,
            "n_columns": n_cols,
            "missing_pct": 0.0,
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "dtype_counts": {},
            "columns": {},
            "is_empty": True,
        }

    missing_pct = round(float(df.isna().mean().mean() * 100), 2) if n_cols else 0.0
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / n_rows) * 100, 2) if n_rows else 0.0

    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()

    per_column = {}
    for col in df.columns:
        col_series = df[col]
        is_num = bool(pd.api.types.is_numeric_dtype(col_series))
        if not is_num:
            # Detect dirty numeric columns (e.g. numbers mixed with 'unknown', '?', 'N/A')
            sample = col_series.dropna().astype(str).str.strip()
            if len(sample) > 0:
                coerced = pd.to_numeric(sample, errors="coerce")
                if (coerced.notna().sum() / len(sample)) >= 0.5:
                    is_num = True

        per_column[col] = {
            "dtype": str(col_series.dtype),
            "missing_pct": round(float(col_series.isna().mean() * 100), 2),
            "n_unique": int(col_series.nunique(dropna=True)),
            "is_numeric": is_num,
            "is_datetime": bool(
                pd.api.types.is_datetime64_any_dtype(col_series)
                or _looks_like_datetime(col_series)
            ),
        }

    return {
        "name": name,
        "n_rows": n_rows,
        "n_columns": n_cols,
        "missing_pct": missing_pct,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "dtype_counts": dtype_counts,
        "columns": per_column,
    }


def _looks_like_datetime(series: pd.Series, sample_size: int = 20) -> bool:
    """
    Heuristic check for string columns that are actually dates (helps
    chart-type rules later).

    Uses is_string_dtype rather than is_object_dtype -- pandas 3.x introduced
    a dedicated string dtype as the default for text columns (previously
    "object"), so is_object_dtype alone no longer catches ordinary string
    columns on pandas 3.x, even though it still works correctly on pandas 2.x.
    """
    if not pd.api.types.is_string_dtype(series):
        return False
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return bool(parsed.notna().mean() > 0.8)
    except Exception:
        return False


def profile_datasets(dataframes: list[pd.DataFrame], names: list[str]) -> dict:
    """
    Profile one or more datasets and produce the aggregate `dataset_profile`
    consumed by the complexity scorer.

    Returns a dict shaped like:
        {
            "datasets": [ per-dataset profile dicts... ],
            "n_datasets": int,
            "n_columns": int (max across datasets, used by scorer),
            "missing_pct": float (average across datasets),
            "duplicate_pct": float (average across datasets),
        }
    """
    if len(dataframes) != len(names):
        raise ValueError("dataframes and names must be the same length")

    individual = [profile_single_dataset(df, name) for df, name in zip(dataframes, names)]

    return {
        "datasets": individual,
        "n_datasets": len(dataframes),
        "n_columns": max((d["n_columns"] for d in individual), default=0),
        "missing_pct": round(float(np.mean([d["missing_pct"] for d in individual])), 2)
        if individual
        else 0.0,
        "duplicate_pct": round(float(np.mean([d["duplicate_pct"] for d in individual])), 2)
        if individual
        else 0.0,
    }
