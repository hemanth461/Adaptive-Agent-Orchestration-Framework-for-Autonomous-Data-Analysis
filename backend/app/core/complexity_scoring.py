"""
Complexity Scoring Engine.

Module: M3 (Complexity Scoring Engine) -- the project's core differentiator.

Hybrid approach:
  1. Deterministic rule-based score (0-10) from four weighted factors.
  2. If the score falls in an ambiguous band near a threshold boundary,
     fall back to a single, narrowly-scoped LLM call to break the tie.

Every call returns a full breakdown so the reason can be logged verbatim
via tracing.log_trace() -- this is what makes routing decisions explainable.
"""
import json

from app.core.llm_client import generate_text


# ---------------------------------------------------------------------------
# Threshold configuration (PRD Section 8). Two ambiguous bands surround the
# Simple/Moderate and Moderate/Complex boundaries -- this is where most
# misclassifications happen in practice, so that's where the LLM tiebreaker
# is spent.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "simple_max": 3.5,
    "moderate_ambiguous_low": 3.5,
    "moderate_ambiguous_high": 4.5,
    "moderate_max": 6.5,
    "complex_ambiguous_low": 6.5,
    "complex_ambiguous_high": 8.0,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_complexity_score(profile: dict, task_operations: list[str]) -> dict:
    """
    Compute the rule-based complexity score (0-10) and its factor breakdown.

    Args:
        profile: the aggregate dataset_profile dict from profiling.profile_datasets()
        task_operations: list of operation strings from task_understanding.parse_user_query()

    Returns:
        {
            "total_score": float,
            "breakdown": {
                "dataset_complexity": float,
                "task_complexity": float,
                "tool_complexity": float,
                "reliability": float,
            }
        }
    """
    n_datasets = profile.get("n_datasets", 1)
    missing_pct = profile.get("missing_pct", 0.0)
    duplicate_pct = profile.get("duplicate_pct", 0.0)
    n_columns = profile.get("n_columns", 0)

    n_viz_ops = sum(
        1
        for op in task_operations
        if op in {"bar_chart", "line_chart", "histogram", "scatter_plot", "heatmap"}
    )
    n_analysis_ops = sum(
        1
        for op in task_operations
        if op
        in {
            "descriptive_stats",
            "correlation_analysis",
            "trend_analysis",
            "group_comparison",
            "distribution_analysis",
        }
    )

    # --- Factor 1: Dataset complexity (weight 0.30) ---
    dataset_score = (
        _clamp01(n_datasets / 3) * 0.4
        + _clamp01(missing_pct / 30) * 0.3
        + _clamp01(duplicate_pct / 20) * 0.15
        + _clamp01(n_columns / 30) * 0.15
    )

    # --- Factor 2: Task complexity (weight 0.35) ---
    task_score = (
        _clamp01(len(task_operations) / 6) * 0.5
        + (1.0 if "multiple_objectives" in task_operations else 0.0) * 0.3
        + _clamp01(n_viz_ops / 4) * 0.2
    )

    # --- Factor 3: Tool/pipeline complexity (weight 0.20) ---
    tool_score = (
        (1.0 if "merge_datasets" in task_operations else 0.0) * 0.4
        + (1.0 if "preprocessing" in task_operations or "clean_missing_values" in task_operations else 0.0) * 0.3
        + (1.0 if n_analysis_ops > 2 else 0.0) * 0.3
    )

    # --- Factor 4: Reliability requirement (weight 0.15) ---
    reliability_score = 1.0 if "statistical_validation" in task_operations else 0.0

    total = (
        dataset_score * 0.30 + task_score * 0.35 + tool_score * 0.20 + reliability_score * 0.15
    ) * 10

    return {
        "total_score": round(total, 2),
        "breakdown": {
            "dataset_complexity": round(dataset_score, 2),
            "task_complexity": round(task_score, 2),
            "tool_complexity": round(tool_score, 2),
            "reliability": round(reliability_score, 2),
        },
    }


def _llm_tiebreaker(
    user_query: str,
    task_operations: list[str],
    profile_summary: str,
    score: float,
    level_a: str,
    level_b: str,
) -> dict:
    """Narrow, single-purpose LLM call used only when the rule score is ambiguous."""
    prompt = f"""Given this task: "{user_query}"
Detected operations: {task_operations}
Dataset profile summary: {profile_summary}
Rule-based score: {score} (borderline between {level_a} and {level_b})

In one sentence, does this task genuinely require independent specialized agents
(e.g. separate reliability checking, multi-step dependent analysis), or can a
single well-tooled agent handle it end-to-end?

Respond with ONLY valid JSON: {{"level": "{level_a}" or "{level_b}", "reason": "..."}}"""

    raw_text = generate_text(prompt=prompt, max_output_tokens=2048)
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(cleaned)
        if result.get("level") not in (level_a, level_b):
            result["level"] = level_a  # fail toward the lower-complexity option
        return result
    except json.JSONDecodeError:
        return {"level": level_a, "reason": "LLM tiebreaker response unparseable; defaulted to lower complexity."}


def assess_complexity(
    profile: dict,
    task_operations: list[str],
    user_query: str = "",
) -> dict:
    """
    Full complexity assessment: rule-based score -> threshold check ->
    conditional LLM tiebreaker -> final level + human-readable reason.

    Returns:
        {
            "total_score": float,
            "breakdown": {...},
            "complexity_level": "simple" | "moderate" | "complex",
            "used_llm_tiebreaker": bool,
            "routing_reason": str,
        }
    """
    scored = compute_complexity_score(profile, task_operations)
    score = scored["total_score"]
    breakdown = scored["breakdown"]

    used_tiebreaker = False
    level: str

    if score <= THRESHOLDS["simple_max"]:
        level = "simple"
    elif THRESHOLDS["moderate_ambiguous_low"] < score <= THRESHOLDS["moderate_ambiguous_high"]:
        used_tiebreaker = True
        profile_summary = (
            f"{profile.get('n_datasets', 1)} dataset(s), "
            f"{profile.get('missing_pct', 0)}% missing, "
            f"{profile.get('n_columns', 0)} columns"
        )
        tb = _llm_tiebreaker(user_query, task_operations, profile_summary, score, "simple", "moderate")
        level = tb["level"]
    elif THRESHOLDS["moderate_ambiguous_high"] < score <= THRESHOLDS["complex_ambiguous_low"]:
        level = "moderate"
    elif THRESHOLDS["complex_ambiguous_low"] < score <= THRESHOLDS["complex_ambiguous_high"]:
        used_tiebreaker = True
        profile_summary = (
            f"{profile.get('n_datasets', 1)} dataset(s), "
            f"{profile.get('missing_pct', 0)}% missing, "
            f"{profile.get('n_columns', 0)} columns"
        )
        tb = _llm_tiebreaker(user_query, task_operations, profile_summary, score, "moderate", "complex")
        level = tb["level"]
    else:
        level = "complex"

    dominant_factor = max(breakdown, key=breakdown.get)
    routing_reason = (
        f"Score: {score}/10 -> {level.capitalize()}. "
        f"Dominant factor: {dominant_factor}={breakdown[dominant_factor]} "
        f"(dataset={breakdown['dataset_complexity']}, task={breakdown['task_complexity']}, "
        f"tool={breakdown['tool_complexity']}, reliability={breakdown['reliability']})."
    )
    if used_tiebreaker:
        routing_reason += " Resolved via LLM tiebreaker (score fell in an ambiguous band)."

    return {
        "total_score": score,
        "breakdown": breakdown,
        "complexity_level": level,
        "used_llm_tiebreaker": used_tiebreaker,
        "routing_reason": routing_reason,
    }
