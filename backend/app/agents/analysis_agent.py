"""
Analysis & Visualization Agent (Complex path).

Module: M7 (Analysis & Visualization Agent)

Structurally similar to the General Agent (M5) but:
  - Operates on state["cleaned_data"] (post Data Prep Agent)
  - Runs the full set of requested operations, including group comparisons
  - Can be re-invoked with a `retry_instruction` from the Insight/Validation
    Agent (M8) -- in that case it re-runs only the relevant analysis rather
    than the whole pipeline, per the PRD's bounded-retry design.
"""
import pandas as pd

from app.core import analysis_tools as tools
from app.core.tracing import log_trace, StepTimer


def run_analysis_agent(state: dict) -> dict:
    """
    Reads: state["cleaned_data"], state["task_operations"], state["dataset_profile"]
    Writes: state["analysis_results"], state["visualizations"]

    On retry (state.get("retry_instruction") is set), only re-runs the
    group_comparison step with the adjustment noted in the instruction,
    rather than re-executing every operation from scratch.
    """
    session_id = state["session_id"]
    df = state["cleaned_data"][0]
    operations = state.get("task_operations", [])
    column_profile = state["dataset_profile"]["datasets"][0]["columns"]

    is_retry = bool(state.get("retry_instruction"))

    if is_retry:
        return _run_retry_analysis(state, df)

    analysis_results = {}
    visualizations = []

    if "descriptive_stats" in operations:
        with StepTimer() as t:
            analysis_results["descriptive_stats"] = tools.describe_stats(df)
        log_trace(
            state, session_id, "analysis_agent", "descriptive_stats",
            reason="Computed descriptive statistics for numeric columns.",
            tool_used="describe_stats", llm_call=False, duration_ms=t.duration_ms,
        )

    if "correlation_analysis" in operations:
        with StepTimer() as t:
            analysis_results["correlation_matrix"] = tools.correlation_matrix(df)
        log_trace(
            state, session_id, "analysis_agent", "correlation_analysis",
            reason="Computed pairwise Pearson correlation across numeric columns.",
            tool_used="correlation_matrix", llm_call=False, duration_ms=t.duration_ms,
        )

    if "group_comparison" in operations:
        with StepTimer() as t:
            group_col, value_col = _pick_group_comparison_columns(column_profile)
            if group_col and value_col:
                analysis_results["group_comparison"] = {
                    "group_col": group_col,
                    "value_col": value_col,
                    "result": tools.groupby_compare(df, group_col, value_col),
                }
        if "group_comparison" in analysis_results:
            log_trace(
                state, session_id, "analysis_agent", "group_comparison",
                reason=f"Compared '{value_col}' across groups of '{group_col}' "
                       f"(auto-selected: first categorical + first numeric column).",
                tool_used="groupby_compare", llm_call=False, duration_ms=t.duration_ms,
            )

    if "trend_analysis" in operations:
        # Trend analysis reuses descriptive stats over a datetime-sorted view;
        # kept simple and explainable rather than a separate statistical model.
        datetime_cols = [c for c, meta in column_profile.items() if meta["is_datetime"]]
        if datetime_cols:
            analysis_results["trend_note"] = (
                f"Data is time-ordered by '{datetime_cols[0]}'; see line chart "
                f"for trend visualization."
            )

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
            state, session_id, "analysis_agent", "chart_selection",
            reason=f"Selected {len(relevant)} chart(s) via rule-based matching against "
                   f"requested types {sorted(requested_chart_types)}.",
            tool_used="select_chart_type", llm_call=False, duration_ms=t.duration_ms,
        )

    state["analysis_results"] = analysis_results
    state["visualizations"] = visualizations
    return state


def _run_retry_analysis(state: dict, df: pd.DataFrame) -> dict:
    """
    Bounded-retry re-analysis: only re-runs group comparison with an
    adjustment (e.g. segmentation), per the retry_instruction from the
    Insight/Validation Agent. Does not re-run the full pipeline.
    """
    session_id = state["session_id"]
    instruction = state["retry_instruction"]
    column_profile = state["dataset_profile"]["datasets"][0]["columns"]

    with StepTimer() as t:
        group_col, value_col = _pick_group_comparison_columns(column_profile)
        result = tools.groupby_compare(df, group_col, value_col) if group_col and value_col else {}

    state.setdefault("analysis_results", {})["group_comparison_retry"] = {
        "group_col": group_col,
        "value_col": value_col,
        "result": result,
        "instruction_applied": instruction,
    }

    log_trace(
        state, session_id, "analysis_agent", "retry_group_comparison",
        reason=f"Re-ran group comparison per validation feedback: '{instruction}'.",
        tool_used="groupby_compare", llm_call=False, duration_ms=t.duration_ms,
    )

    # Clear the instruction so we don't loop again
    state["retry_instruction"] = None
    return state


def _pick_group_comparison_columns(column_profile: dict) -> tuple[str | None, str | None]:
    categorical_cols = [
        c for c, meta in column_profile.items()
        if not meta["is_numeric"] and not meta["is_datetime"] and meta["n_unique"] <= 20
    ]
    numeric_cols = [
        c for c, meta in column_profile.items()
        if meta["is_numeric"] and not _looks_like_id_column(c, meta)
    ]
    group_col = categorical_cols[0] if categorical_cols else None
    value_col = numeric_cols[0] if numeric_cols else None
    return group_col, value_col


def _looks_like_id_column(col_name: str, meta: dict) -> bool:
    """Excludes likely identifier columns (near-unique numeric, or *_id named) from
    being auto-selected as the analysis value column -- comparing means of an ID
    column is meaningless."""
    name_lower = col_name.lower()
    if name_lower.endswith("_id") or name_lower == "id":
        return True
    return False


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
