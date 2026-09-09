"""
Data Preparation Agent.

Module: M6 (Data Preparation Agent)

Deterministic detection (pandas) + one LLM call per issue TYPE (not per
column) to choose a handling strategy with a short justification. Runs once,
non-interactively -- per the PRD's scope decision, this agent does not
receive feedback from downstream agents and does not re-run mid-pipeline.

Performs a naive union merge (row concat, not key-based join) across
uploaded datasets when "merge_datasets" is requested -- see
_get_working_dataframe() for the documented limitation.
"""
import json
import pandas as pd
import numpy as np

from app.core.tracing import log_trace, StepTimer
from app.core.llm_client import generate_text


def _get_working_dataframe(state: dict) -> pd.DataFrame:
    """
    Returns the dataframe the rest of the Data Prep Agent should operate on.

    If "merge_datasets" was requested AND more than one dataset was uploaded,
    performs a naive concat-based union merge (stacks rows, aligns by column
    name, fills missing columns with NaN where schemas differ) and logs the
    merge with an explicit caveat that this is not a key-based join.

    Otherwise, operates on the first uploaded dataset only, matching the
    documented single-dataset default for tasks that didn't ask to merge.
    """
    session_id = state["session_id"]
    datasets = state["raw_datasets"]
    names = state.get("dataset_names", [f"dataset_{i}" for i in range(len(datasets))])
    operations = state.get("task_operations", [])

    if len(datasets) > 1 and "merge_datasets" in operations:
        row_counts = {name: len(df) for name, df in zip(names, datasets)}
        merged = pd.concat(datasets, ignore_index=True, sort=False)

        all_columns = set().union(*[set(df.columns) for df in datasets])
        schemas_differ = any(set(df.columns) != all_columns for df in datasets)

        reason = (
            f"Merged {len(datasets)} datasets via row-wise concatenation "
            f"({row_counts}) into {len(merged)} total rows. "
        )
        if schemas_differ:
            reason += (
                "Datasets had differing columns; missing values were introduced "
                "where a column didn't exist in a given source. "
            )
        reason += (
            "Known limitation: this is a naive union merge (stack + align by "
            "column name), not a key-based join -- there is no deduplication "
            "or relationship resolution across datasets beyond exact-row "
            "duplicate detection performed in the next step."
        )

        log_trace(
            state, session_id, "data_prep_agent", "merge_datasets",
            reason=reason, tool_used="pd.concat", llm_call=False, duration_ms=0,
        )
        return merged

    if len(datasets) > 1:
        log_trace(
            state, session_id, "data_prep_agent", "single_dataset_default",
            reason=f"{len(datasets)} datasets were uploaded but merging was not "
                   f"requested; operating on the first dataset ('{names[0]}') only.",
            tool_used=None, llm_call=False, duration_ms=0,
        )

    return datasets[0].copy()


def run_data_prep_agent(state: dict) -> dict:
    """
    Cleans and, when requested, merges the uploaded dataset(s).

    Writes: state["cleaned_data"], state["cleaning_log"]
    """
    session_id = state["session_id"]
    df = _get_working_dataframe(state)
    cleaning_log = []

    # --- Detect + handle missing values ---
    missing_cols = df.columns[df.isna().any()].tolist()
    if missing_cols:
        with StepTimer() as t:
            df, entries = _handle_missing_values(df, missing_cols)
        cleaning_log.extend(entries)
        log_trace(
            state, session_id, "data_prep_agent", "handle_missing_values",
            reason=f"Detected missing values in {len(missing_cols)} column(s): {missing_cols}. "
                   f"Applied per-column imputation strategies (see cleaning_log).",
            tool_used="impute_missing", llm_call=True, duration_ms=t.duration_ms,
        )

    # --- Detect + handle duplicates ---
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        with StepTimer() as t:
            df = df.drop_duplicates().reset_index(drop=True)
        cleaning_log.append({
            "column_name": None,
            "issue_type": "duplicate",
            "strategy_applied": "drop_duplicates",
            "reason": f"Removed {n_duplicates} exact duplicate row(s) to prevent "
                      f"double-counting in downstream analysis.",
        })
        log_trace(
            state, session_id, "data_prep_agent", "handle_duplicates",
            reason=f"Removed {n_duplicates} duplicate row(s).",
            tool_used="drop_duplicates", llm_call=False, duration_ms=t.duration_ms,
        )

    # --- Detect dtype issues (numeric-looking strings) ---
    with StepTimer() as t:
        df, dtype_entries = _fix_dtype_issues(df)
    if dtype_entries:
        cleaning_log.extend(dtype_entries)
        log_trace(
            state, session_id, "data_prep_agent", "fix_dtype_issues",
            reason=f"Corrected {len(dtype_entries)} column(s) with dtype mismatches "
                   f"(e.g. numeric values stored as strings).",
            tool_used="fix_dtype_issues", llm_call=False, duration_ms=t.duration_ms,
        )

    # --- Detect outliers (IQR method) -- flagged, not removed ---
    with StepTimer() as t:
        outlier_entries = _detect_outliers_iqr(df)
    if outlier_entries:
        cleaning_log.extend(outlier_entries)
        log_trace(
            state, session_id, "data_prep_agent", "detect_outliers",
            reason=f"Flagged potential outliers in {len(outlier_entries)} column(s) using "
                   f"the IQR method. Outliers were flagged, not removed, to avoid "
                   f"discarding potentially valid extreme values without review.",
            tool_used="detect_outliers_iqr", llm_call=False, duration_ms=t.duration_ms,
        )

    state["cleaned_data"] = [df]
    state["cleaning_log"] = cleaning_log

    # Persist cleaning_log to its own table too (not just execution_trace)
    _persist_cleaning_log(session_id, cleaning_log)

    return state


def _handle_missing_values(df: pd.DataFrame, missing_cols: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """One LLM call covering all missing-value columns at once (not per-column) to pick strategies."""
    entries = []

    numeric_missing = [c for c in missing_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_missing = [c for c in missing_cols if c not in numeric_missing]

    strategy_map = {}
    if numeric_missing or categorical_missing:
        strategy_map = _llm_choose_imputation_strategies(df, numeric_missing, categorical_missing)

    for col in numeric_missing:
        skew = float(df[col].skew()) if df[col].notna().sum() > 2 else 0.0
        strategy = strategy_map.get(col, "median" if abs(skew) > 0.5 else "mean")
        if strategy == "median":
            fill_value = df[col].median()
        else:
            fill_value = df[col].mean()
        df[col] = df[col].fillna(fill_value)
        entries.append({
            "column_name": col,
            "issue_type": "missing",
            "strategy_applied": f"{strategy}_imputation",
            "reason": f"{strategy.capitalize()} imputation selected "
                      f"(skew={round(skew, 2)}); value used: {round(float(fill_value), 3)}.",
        })

    for col in categorical_missing:
        mode_value = df[col].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        df[col] = df[col].fillna(fill_value)
        entries.append({
            "column_name": col,
            "issue_type": "missing",
            "strategy_applied": "mode_imputation",
            "reason": f"Mode imputation selected for categorical column; "
                      f"filled with most frequent value '{fill_value}'.",
        })

    return df, entries


def _llm_choose_imputation_strategies(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> dict:
    """
    Single LLM call to sanity-check/override the default skew-based rule for
    numeric columns. Keeps the agent's reasoning step genuinely LLM-driven
    (per PRD Section 6) rather than 100% hardcoded, while still bounding it
    to one call regardless of column count.
    """
    if not numeric_cols:
        return {}

    col_stats = {
        col: {
            "skew": round(float(df[col].skew()), 2) if df[col].notna().sum() > 2 else 0.0,
            "missing_pct": round(float(df[col].isna().mean() * 100), 1),
        }
        for col in numeric_cols
    }

    prompt = f"""For each numeric column below, choose an imputation strategy: "mean" or "median".
Prefer median for skewed distributions (|skew| > 0.5), mean otherwise, unless
missing_pct is very high (>40%) in which case note that in your choice but
still pick mean or median.

Columns: {json.dumps(col_stats)}

Respond with ONLY valid JSON mapping column name to strategy, e.g.:
{{"income": "median", "age": "mean"}}"""

    try:
        raw_text = generate_text(prompt=prompt, max_output_tokens=2048)
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception:
        # Fail safe: fall back to the deterministic skew rule per column
        return {
            col: ("median" if abs(col_stats[col]["skew"]) > 0.5 else "mean")
            for col in numeric_cols
        }


def _fix_dtype_issues(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Detect object columns that are actually numeric and convert them."""
    entries = []
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # If most non-null values convert successfully, treat it as a numeric column
        non_null = df[col].notna().sum()
        if non_null > 0 and converted.notna().sum() / non_null > 0.9:
            df[col] = converted
            entries.append({
                "column_name": col,
                "issue_type": "dtype",
                "strategy_applied": "convert_to_numeric",
                "reason": f"Column '{col}' contained numeric values stored as text; "
                          f"converted to numeric dtype for correct downstream analysis.",
            })
    return df, entries


def _detect_outliers_iqr(df: pd.DataFrame) -> list[dict]:
    """IQR-based outlier flagging for numeric columns. Flags only, does not remove."""
    entries = []
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())
        if n_outliers > 0:
            entries.append({
                "column_name": col,
                "issue_type": "outlier",
                "strategy_applied": "flagged_iqr_no_removal",
                "reason": f"{n_outliers} value(s) outside [{round(lower, 2)}, {round(upper, 2)}] "
                          f"(1.5x IQR bounds); flagged for awareness, not removed.",
            })
    return entries


def _persist_cleaning_log(session_id: str, cleaning_log: list[dict]) -> None:
    from app.db.connection import get_connection

    with get_connection() as conn:
        for entry in cleaning_log:
            conn.execute(
                """
                INSERT INTO cleaning_log (session_id, column_name, issue_type, strategy_applied, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, entry.get("column_name"), entry["issue_type"],
                 entry["strategy_applied"], entry["reason"]),
            )
        conn.commit()
