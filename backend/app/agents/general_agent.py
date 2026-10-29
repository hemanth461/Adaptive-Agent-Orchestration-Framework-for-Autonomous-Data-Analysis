"""
General Agent.

Module: M5 (General Agent, Simple/Moderate paths)

For SIMPLE tasks: runs a minimal set of requested operations directly.
For MODERATE tasks: same agent, but treated as "tool-augmented" -- it has
access to the full analysis_tools toolkit and runs whatever combination of
tools the task_operations call for, without invoking the specialized
multi-agent chain (no Insight/Validation agent involved).

Includes lightweight, deterministic cleaning (missing-value imputation via
skew-based mean/median, duplicate removal) when "clean_missing_values" or
"clean_duplicates" is requested -- this matches the PRD's own Moderate
example ("Clean minor missing values, analyze correlations, and create
multiple charts") without pulling in the full Data Prep Agent's per-issue
LLM strategy call, keeping this path cheap as intended.
"""
import pandas as pd
import numpy as np

from app.core import analysis_tools as tools
from app.core.tracing import log_trace, StepTimer


def run_general_agent(state: dict) -> dict:
    """
    Execute the Simple/Moderate single-agent path.

    Reads: state["raw_datasets"], state["task_operations"], state["dataset_profile"]
    Writes: state["analysis_results"], state["visualizations"], state["cleaning_log"]
    """
    session_id = state["session_id"]
    df = _get_primary_dataframe(state)
    operations = state.get("task_operations", [])
    cleaning_log = []

    # --- Lightweight deterministic cleaning (no LLM -- keeps Moderate cheap) ---
    if "clean_missing_values" in operations:
        with StepTimer() as t:
            df, entries = _clean_missing_values_simple(df)
        cleaning_log.extend(entries)
        if entries:
            log_trace(
                state, session_id, "general_agent", "clean_missing_values",
                reason=f"Applied deterministic mean/median imputation to "
                       f"{len(entries)} column(s) based on skew, without an "
                       f"LLM strategy call (kept lightweight for this task level).",
                tool_used="clean_missing_values_simple", llm_call=False,
                duration_ms=t.duration_ms,
            )

    if "clean_duplicates" in operations:
        n_before = len(df)
        with StepTimer() as t:
            df = df.drop_duplicates().reset_index(drop=True)
        n_removed = n_before - len(df)
        if n_removed > 0:
            cleaning_log.append({
                "column_name": None,
                "issue_type": "duplicate",
                "strategy_applied": "drop_duplicates",
                "reason": f"Removed {n_removed} exact duplicate row(s).",
            })
            log_trace(
                state, session_id, "general_agent", "clean_duplicates",
                reason=f"Removed {n_removed} duplicate row(s).",
                tool_used="drop_duplicates", llm_call=False, duration_ms=t.duration_ms,
            )

    column_profile = state["dataset_profile"]["datasets"][0]["columns"]
    for col in df.columns:
        if col in column_profile:
            column_profile[col]["is_numeric"] = bool(pd.api.types.is_numeric_dtype(df[col]))
            column_profile[col]["dtype"] = str(df[col].dtype)
            column_profile[col]["missing_pct"] = round(float(df[col].isna().mean() * 100), 2)

    analysis_results = {}
    visualizations = []

    # --- Statistics operations ---
    if "descriptive_stats" in operations:
        with StepTimer() as t:
            analysis_results["descriptive_stats"] = tools.describe_stats(df)
        log_trace(
            state, session_id, "general_agent", "descriptive_stats",
            reason="Computed descriptive statistics for numeric columns as requested.",
            tool_used="describe_stats", llm_call=False, duration_ms=t.duration_ms,
        )

    if "correlation_analysis" in operations:
        with StepTimer() as t:
            analysis_results["correlation_matrix"] = tools.correlation_matrix(df)
        log_trace(
            state, session_id, "general_agent", "correlation_analysis",
            reason="Computed pairwise Pearson correlation across numeric columns.",
            tool_used="correlation_matrix", llm_call=False, duration_ms=t.duration_ms,
        )

    if "group_comparison" in operations:
        with StepTimer() as t:
            cat_cols = [c for c, m in column_profile.items() if not m.get("is_numeric") and m.get("n_unique", 99) <= 20]
            num_cols = [c for c, m in column_profile.items() if m.get("is_numeric") and not (c.lower().endswith("_id") or c.lower() == "id")]
            if cat_cols and num_cols:
                analysis_results["group_comparison"] = {
                    "group_col": cat_cols[0],
                    "value_col": num_cols[0],
                    "result": tools.groupby_compare(df, cat_cols[0], num_cols[0]),
                }
        if "group_comparison" in analysis_results:
            log_trace(
                state, session_id, "general_agent", "group_comparison",
                reason=f"Compared '{num_cols[0]}' across groups of '{cat_cols[0]}'.",
                tool_used="groupby_compare", llm_call=False, duration_ms=t.duration_ms,
            )

    # --- Visualization operations: use rule-based chart selection, then filter
    #     to only what the user actually asked for (or all recommendations if
    #     the task is broad, e.g. "create visualizations") ---
    requested_chart_types = {
        op for op in operations
        if op in {"bar_chart", "line_chart", "histogram", "scatter_plot", "heatmap"}
    }

    if requested_chart_types:
        with StepTimer() as t:
            recommendations = tools.select_chart_type(df, column_profile)
        relevant = [r for r in recommendations if r["chart_type"] in requested_chart_types]

        for rec in relevant:
            chart_result = _execute_chart(df, rec, session_id)
            if chart_result:
                chart_result["reason"] = rec["rule_reason"]
                visualizations.append(chart_result)

        log_trace(
            state, session_id, "general_agent", "chart_selection",
            reason=f"Selected {len(relevant)} chart(s) via rule-based matching against "
                   f"requested types {sorted(requested_chart_types)}.",
            tool_used="select_chart_type", llm_call=False, duration_ms=t.duration_ms,
        )

    state["cleaned_data"] = [df]
    state["analysis_results"] = analysis_results
    state["visualizations"] = visualizations
    state["cleaning_log"] = cleaning_log
    return state


def _get_primary_dataframe(state: dict) -> pd.DataFrame:
    """Simple/Moderate paths operate on a single dataset (the first uploaded one)."""
    return state["raw_datasets"][0]


def _clean_missing_values_simple(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Deterministic dirty numeric coercion and mean/median imputation based on skew,
    with no LLM call -- appropriate for the Moderate task level.
    """
    entries = []
    df = df.copy()

    # Step 1: Detect and coerce dirty numeric columns (where >= 50% non-null values are numbers)
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna()
        if len(sample) > 0:
            coerced = pd.to_numeric(sample, errors="coerce")
            if (coerced.notna().sum() / len(sample)) >= 0.5:
                dirty_count = int((sample.notna() & coerced.isna()).sum())
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if dirty_count > 0:
                    entries.append({
                        "column_name": col,
                        "issue_type": "invalid_values",
                        "strategy_applied": "coerce_to_numeric",
                        "reason": f"Coerced {dirty_count} invalid non-numeric text values to NaN in numeric column '{col}'.",
                    })

    # Step 2: Impute missing values
    missing_cols = df.columns[df.isna().any()].tolist()

    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            skew = float(df[col].skew()) if df[col].notna().sum() > 2 else 0.0
            strategy = "median" if abs(skew) > 0.5 else "mean"
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)
            entries.append({
                "column_name": col,
                "issue_type": "missing",
                "strategy_applied": f"{strategy}_imputation",
                "reason": f"{strategy.capitalize()} imputation selected "
                          f"(skew={round(skew, 2)}); value used: {round(float(fill_value), 3)}.",
            })
        else:
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


def _execute_chart(df: pd.DataFrame, rec: dict, session_id: str) -> dict | None:
    chart_type = rec["chart_type"]
    cols = rec["columns"]
    try:
        if chart_type == "line_chart":
            return tools.plot_line(df, cols[0], cols[1], session_id)
        if chart_type == "bar_chart":
            return tools.plot_bar(df, cols[0], cols[1], session_id)
        if chart_type == "scatter_plot":
            return tools.plot_scatter(df, cols[0], cols[1], session_id)
        if chart_type == "histogram":
            return tools.plot_histogram(df, cols[0], session_id)
        if chart_type == "heatmap":
            return tools.plot_heatmap(df, session_id)
    except Exception:
        return None
    return None
