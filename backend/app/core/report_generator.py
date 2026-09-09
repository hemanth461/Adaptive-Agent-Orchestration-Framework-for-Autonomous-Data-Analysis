"""
Report Generation Module.

Module: M10 (Report Generation)

Mostly templated: pulls structured data straight from the final AgentState
and formats it into Markdown. Only ONE LLM call is used, to write a short
connective/executive-summary paragraph -- the rest is deterministic string
formatting so the report is fast, cheap, and never invents numbers.
"""
import json

from app.core.tracing import log_trace, StepTimer
from app.core.llm_client import generate_text


def generate_report(state: dict) -> str:
    """
    Builds the final Markdown report from state. Writes state["final_report"]
    and logs the step. Safe to call for any complexity level (Simple/Moderate
    paths simply have empty cleaning_log/insights sections).
    """
    session_id = state["session_id"]

    with StepTimer() as t:
        exec_summary = _generate_executive_summary(state)
        report_md = _assemble_report(state, exec_summary)

    state["final_report"] = report_md
    log_trace(
        state, session_id, "report_generator", "generate_final_report",
        reason="Assembled final report from structured state (templated sections) "
               "with one LLM call for the executive summary paragraph.",
        tool_used=None, llm_call=True, duration_ms=t.duration_ms,
    )
    return report_md


def _generate_executive_summary(state: dict) -> str:
    """Single LLM call: 2-3 sentence plain-language summary of what was found."""
    context = {
        "user_query": state.get("user_query", ""),
        "complexity_level": state.get("complexity_level", ""),
        "n_insights": len(state.get("insights", [])),
        "n_supported": sum(1 for v in state.get("validation_results", []) if v.get("supported")),
        "n_charts": len(state.get("visualizations", [])),
        "cleaning_actions": len(state.get("cleaning_log", [])),
    }

    prompt = f"""Write a 2-3 sentence executive summary for a data analysis report.
Context (JSON, use only these facts, do not invent numbers): {json.dumps(context)}

Plain language, no headers, no markdown formatting, just the paragraph text."""

    try:
        return generate_text(prompt, max_output_tokens=2048)
    except Exception:
        return (
            f"This report covers a {context['complexity_level']} analysis task. "
            f"{context['n_insights']} insight(s) were generated, "
            f"{context['n_supported']} of which were statistically supported. "
            f"{context['n_charts']} visualization(s) were produced."
        )


def _assemble_report(state: dict, exec_summary: str) -> str:
    sections = []

    sections.append(f"# Data Analysis Report\n")
    sections.append(f"**Request:** {state.get('user_query', '')}\n")
    sections.append(f"## Executive Summary\n{exec_summary}\n")

    # --- Architecture decision ---
    sections.append("## Architecture Decision\n")
    sections.append(
        f"- **Complexity Score:** {state.get('complexity_score', 'N/A')}/10\n"
        f"- **Level Selected:** {state.get('complexity_level', 'N/A').capitalize()}\n"
        f"- **Reason:** {state.get('routing_reason', 'N/A')}\n"
        f"- **LLM Tiebreaker Used:** {state.get('used_llm_tiebreaker', False)}\n"
    )

    # --- Dataset overview ---
    profile = state.get("dataset_profile", {})
    if profile:
        sections.append("## Dataset Overview\n")
        for ds in profile.get("datasets", []):
            sections.append(
                f"- **{ds['name']}**: {ds['n_rows']} rows x {ds['n_columns']} columns, "
                f"{ds['missing_pct']}% missing, {ds['duplicate_count']} duplicate rows\n"
            )

    # --- Data quality / cleaning ---
    cleaning_log = state.get("cleaning_log", [])
    if cleaning_log:
        sections.append("## Data Quality & Preprocessing\n")
        for entry in cleaning_log:
            col = entry.get("column_name") or "(table-level)"
            sections.append(f"- **{col}** [{entry['issue_type']}]: {entry['reason']}\n")

    # --- Analysis results ---
    analysis_results = state.get("analysis_results", {})
    if analysis_results:
        sections.append("## Analysis Results\n")
        if "descriptive_stats" in analysis_results:
            sections.append("**Descriptive Statistics:**\n")
            for col, stats in analysis_results["descriptive_stats"].items():
                sections.append(f"- {col}: mean={stats['mean']}, median={stats['median']}, std={stats['std']}\n")
        if "group_comparison" in analysis_results:
            gc = analysis_results["group_comparison"]
            sections.append(f"\n**Group Comparison** ({gc['value_col']} by {gc['group_col']}):\n")
            for group, vals in gc["result"].items():
                sections.append(f"- {group}: mean={vals.get('mean')}, n={vals.get('count')}\n")

    # --- Visualizations ---
    visualizations = state.get("visualizations", [])
    if visualizations:
        sections.append("\n## Visualizations\n")
        for v in visualizations:
            sections.append(f"- **{v['type']}** ({', '.join(v['columns'])}): {v['reason']}\n")

    # --- Insights & validation ---
    validation_results = state.get("validation_results", [])
    if validation_results:
        sections.append("\n## Insights & Validation\n")
        for v in validation_results:
            status = "✅ Supported" if v["supported"] else "⚠️ Unsupported (caveat)"
            sections.append(f"- **{v['claim']}** — {status}\n  - {v['reason']}\n")

    # --- Execution trace summary ---
    trace = state.get("execution_trace", [])
    if trace:
        n_llm_calls = sum(1 for t in trace if t.get("llm_call"))
        sections.append(
            f"\n## Execution Summary\n"
            f"- Total steps: {len(trace)}\n"
            f"- LLM calls made: {n_llm_calls}\n"
            f"- Retry count: {state.get('retry_count', 0)}\n"
        )

    return "\n".join(sections)
