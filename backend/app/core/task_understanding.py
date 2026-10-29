"""
Task Understanding Module.

Module: M2 (Task Understanding)

Single LLM call that turns the user's free-text analysis request into a
structured list of operations the rest of the system can reason about
(used by the Complexity Scorer, the Analysis Agent's chart-selection rules,
and the Insight/Validation Agent's test-trigger table).

Keep this to ONE call per session -- do not re-parse the query multiple times.
"""
import json

from app.core.llm_client import generate_text

# The controlled vocabulary the rest of the system understands. The LLM is
# instructed to only choose from this list so downstream rule-based logic
# (chart selection, test selection, complexity scoring) stays deterministic.
KNOWN_OPERATIONS = [
    "clean_missing_values",
    "clean_duplicates",
    "preprocessing",
    "merge_datasets",
    "descriptive_stats",
    "correlation_analysis",
    "trend_analysis",
    "group_comparison",
    "distribution_analysis",
    "bar_chart",
    "line_chart",
    "histogram",
    "scatter_plot",
    "heatmap",
    "statistical_validation",
    "multiple_objectives",
]

SYSTEM_PROMPT = f"""You convert a user's data-analysis request into a structured JSON object.

Only use operations from this fixed vocabulary: {json.dumps(KNOWN_OPERATIONS)}

Rules:
- Include "multiple_objectives" if the request asks for more than one distinct kind of analysis outcome (e.g. cleaning AND trends AND validation).
- Include "statistical_validation" only if the user explicitly or implicitly asks to verify/validate/confirm a claim statistically.
- Infer chart operations (bar_chart, line_chart, etc.) only if a visualization is requested or clearly implied.
- Respond with ONLY valid JSON, no preamble, no markdown fences.

Output schema:
{{"operations": ["op1", "op2", ...], "summary": "one sentence describing the task"}}
"""


def _heuristic_parse_query(query: str) -> dict:
    """Fallback keyword-based parser when LLM is unavailable or offline."""
    q = query.lower()
    ops = []

    if any(k in q for k in ["clean", "missing", "imput", "null", "nan"]):
        ops.append("clean_missing_values")
    if any(k in q for k in ["duplicate", "dedup", "unique"]):
        ops.append("clean_duplicates")
    if any(k in q for k in ["merge", "join", "combine", "concat"]):
        ops.append("merge_datasets")
    if any(k in q for k in ["stat", "describ", "summar", "average", "mean", "median", "overview"]):
        ops.append("descriptive_stats")
    if any(k in q for k in ["correlat", "relation", "association"]):
        ops.append("correlation_analysis")
    if any(k in q for k in ["trend", "over time", "time series", "progression"]):
        ops.append("trend_analysis")
    if any(k in q for k in ["group", "compare", "by region", "by department", "by category", "comparison"]):
        ops.append("group_comparison")
    if any(k in q for k in ["distribut", "spread", "range"]):
        ops.append("distribution_analysis")
    if "bar" in q:
        ops.append("bar_chart")
    if "line" in q:
        ops.append("line_chart")
    if "hist" in q:
        ops.append("histogram")
    if "scatter" in q:
        ops.append("scatter_plot")
    if "heatmap" in q:
        ops.append("heatmap")
    if any(k in q for k in ["validat", "significan", "hypothes", "p-value", "t-test", "anova", "confirm statistically"]):
        ops.append("statistical_validation")

    # If general visualization requested without specific chart type
    if any(k in q for k in ["chart", "plot", "graph", "visual", "visualize"]) and not any(
        c in ops for c in ["bar_chart", "line_chart", "histogram", "scatter_plot", "heatmap"]
    ):
        ops.append("bar_chart")

    if len(ops) >= 2:
        ops.append("multiple_objectives")

    if not ops:
        ops.append("descriptive_stats")

    return {
        "operations": ops,
        "summary": f"Interpreted {len(ops)} target operation(s) from query.",
    }


def parse_user_query(user_query: str) -> dict:
    """
    Parse the user's natural-language request into structured task_operations.

    Returns:
        {"operations": [...], "summary": "..."}
    """
    try:
        raw_text = generate_text(
            prompt=user_query,
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
        )
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
    except Exception:
        parsed = _heuristic_parse_query(user_query)

    # Filter to known vocabulary only, in case the model hallucinates an operation
    parsed["operations"] = [op for op in parsed.get("operations", []) if op in KNOWN_OPERATIONS]

    # Edge case: if nothing recognizable was parsed (e.g. a vague or off-topic
    # query), fall back to a minimal default rather than returning an empty
    # pipeline that produces no output at all.
    if not parsed["operations"]:
        parsed["operations"] = ["descriptive_stats"]
        parsed["summary"] = parsed.get("summary") or (
            "No specific operations detected in the request; defaulting to "
            "descriptive statistics."
        )

    return parsed
