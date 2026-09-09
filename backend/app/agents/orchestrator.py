"""
Adaptive Orchestrator.

Module: M4 (Adaptive Orchestrator)

Not a chatty LLM agent -- a routing supervisor. Runs three deterministic-ish
steps (profiling is pure pandas; task parsing and the complexity tiebreaker
are the only LLM calls) and writes the routing decision + reason to the trace.
"""
from app.core.profiling import profile_datasets
from app.core.task_understanding import parse_user_query
from app.core.complexity_scoring import assess_complexity
from app.core.tracing import log_trace, StepTimer


def run_orchestrator(state: dict) -> dict:
    """
    Entry node of the graph. Populates dataset_profile, task_operations,
    complexity_score/level, and routing_reason on the shared state.

    This function's return value's `complexity_level` field is what the
    LangGraph conditional edge reads to decide which branch to take next.
    """
    session_id = state["session_id"]

    # --- Step 1: Dataset profiling (no LLM) ---
    with StepTimer() as t:
        profile = profile_datasets(state["raw_datasets"], state["dataset_names"])
    state["dataset_profile"] = profile
    log_trace(
        state, session_id, "orchestrator", "dataset_profiling",
        reason=f"Profiled {profile['n_datasets']} dataset(s): "
               f"{profile['n_columns']} max columns, {profile['missing_pct']}% avg missing, "
               f"{profile['duplicate_pct']}% avg duplicates.",
        tool_used="profile_datasets", llm_call=False, duration_ms=t.duration_ms,
    )

    # --- Step 2: Task understanding (1 LLM call) ---
    with StepTimer() as t:
        parsed = parse_user_query(state["user_query"])
    state["task_operations"] = parsed["operations"]
    log_trace(
        state, session_id, "orchestrator", "task_understanding",
        reason=f"Parsed query into operations: {parsed['operations']}. "
               f"Summary: {parsed.get('summary', '')}",
        tool_used=None, llm_call=True, duration_ms=t.duration_ms,
    )

    # --- Step 3: Complexity assessment (rule-based + conditional LLM tiebreaker) ---
    with StepTimer() as t:
        assessment = assess_complexity(profile, state["task_operations"], state["user_query"])
    state["complexity_score"] = assessment["total_score"]
    state["complexity_breakdown"] = assessment["breakdown"]
    state["complexity_level"] = assessment["complexity_level"]
    state["used_llm_tiebreaker"] = assessment["used_llm_tiebreaker"]
    state["routing_reason"] = assessment["routing_reason"]

    log_trace(
        state, session_id, "orchestrator", "complexity_assessment",
        reason=assessment["routing_reason"],
        tool_used="assess_complexity",
        llm_call=assessment["used_llm_tiebreaker"],
        duration_ms=t.duration_ms,
    )

    # --- Forced-mode override (used only by the Phase 5 evaluation harness to
    #     compare architectures on identical input). The naturally-computed
    #     level/score above is preserved as `adaptive_recommended_level` so
    #     the comparison can show what adaptive routing *would* have chosen,
    #     even when a different mode is forced for the run. ---
    forced_mode = state.get("_forced_mode")
    if forced_mode:
        state["adaptive_recommended_level"] = state["complexity_level"]
        forced_level_map = {"single": "simple", "multi": "complex", "adaptive": state["complexity_level"]}
        overridden_level = forced_level_map.get(forced_mode, state["complexity_level"])
        state["complexity_level"] = overridden_level
        state["routing_reason"] = (
            f"[FORCED MODE: {forced_mode}] Overriding adaptively-recommended level "
            f"'{state['adaptive_recommended_level']}' with '{overridden_level}' for evaluation comparison."
        )
        log_trace(
            state, session_id, "orchestrator", "forced_mode_override",
            reason=state["routing_reason"],
            tool_used=None, llm_call=False, duration_ms=0,
        )

    log_trace(
        state, session_id, "orchestrator", "routing_decision",
        reason=f"Routing to '{state['complexity_level'].upper()}' path.",
        tool_used=None, llm_call=False, duration_ms=0,
    )

    return state


def route_by_complexity(state: dict) -> str:
    """
    Conditional edge function for LangGraph: returns the name of the next
    node/branch based on state["complexity_level"].
    """
    level = state.get("complexity_level", "simple")
    if level == "complex":
        return "complex_path"
    # Both "simple" and "moderate" use the same General Agent node;
    # the distinction is which tools get invoked (driven by task_operations),
    # not a different node.
    return "single_agent_path"
