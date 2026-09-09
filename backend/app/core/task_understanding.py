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


def parse_user_query(user_query: str) -> dict:
    """
    Parse the user's natural-language request into structured task_operations.

    Returns:
        {"operations": [...], "summary": "..."}
    """
    raw_text = generate_text(
        prompt=user_query,
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=2048,
    )

    # Defensive parsing in case the model wraps output in fences despite instructions
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fail safe: treat as a single generic operation rather than crashing the pipeline
        parsed = {
            "operations": ["descriptive_stats"],
            "summary": user_query[:200],
        }

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
