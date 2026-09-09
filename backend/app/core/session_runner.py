"""
Session Runner.

Bridges the FastAPI layer and the LangGraph pipeline: takes uploaded files +
a query, builds the initial AgentState, invokes the graph, and persists the
sessions row. Also handles the "forced_mode" override used by the evaluation
harness (M13, Phase 5) to force single/multi architecture regardless of the
computed complexity score.
"""
import json
import uuid
import time
import pandas as pd

from app.core.graph import get_graph
from app.core.profiling import profile_datasets
from app.core.task_understanding import parse_user_query
from app.core.complexity_scoring import assess_complexity
from app.db.connection import get_connection


def run_session(
    dataframes: list[pd.DataFrame],
    dataset_names: list[str],
    user_query: str,
    forced_mode: str | None = None,
) -> dict:
    """
    Run one full analysis session.

    Args:
        dataframes: uploaded datasets, already parsed into pandas
        dataset_names: matching display names
        user_query: the user's analysis request
        forced_mode: None for normal adaptive routing, or "simple"/"moderate"/
                     "complex" to force a specific architecture (used by the
                     Phase 5 evaluation harness to compare architectures on
                     the same input)

    Returns:
        The final AgentState dict (includes final_report, execution_trace, etc.)
    """
    session_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    initial_state = {
        "session_id": session_id,
        "raw_datasets": dataframes,
        "dataset_names": dataset_names,
        "user_query": user_query,
        "execution_trace": [],
        "retry_count": 0,
    }

    if forced_mode:
        # Wired into orchestrator.run_orchestrator: the naturally-computed
        # level is preserved as adaptive_recommended_level, and the graph
        # routes according to the forced level instead. This is what lets
        # the evaluation harness compare single/multi/adaptive on identical input.
        initial_state["_forced_mode"] = forced_mode

    graph = get_graph()
    result = graph.invoke(initial_state)

    elapsed = time.perf_counter() - start_time
    _persist_session(session_id, result, forced_mode, elapsed)

    if forced_mode:
        _persist_eval_run(session_id, result, forced_mode, elapsed)

    return result


def _compute_metrics(state: dict, elapsed_sec: float) -> dict:
    """
    Derive the efficiency/cost metrics used by the evaluation harness (PRD
    Section 13). Token/cost figures are estimates, not exact API accounting --
    documented as such rather than presented as precise billing data.
    """
    trace = state.get("execution_trace", [])
    llm_call_count = sum(1 for t in trace if t.get("llm_call"))

    # Rough estimate: ~600 tokens per LLM call on average (prompt + completion)
    # for this system's call sizes (query parsing, imputation strategy,
    # insight generation, executive summary). This is a coarse approximation
    # for comparative purposes only, not a substitute for real API usage logs.
    AVG_TOKENS_PER_CALL = 600
    COST_PER_1K_TOKENS = 0.003  # approximate blended rate, adjust to actual pricing

    estimated_tokens = llm_call_count * AVG_TOKENS_PER_CALL
    estimated_cost = round((estimated_tokens / 1000) * COST_PER_1K_TOKENS, 5)

    task_completed = bool(state.get("final_report"))

    return {
        "execution_time_sec": round(elapsed_sec, 3),
        "llm_call_count": llm_call_count,
        "estimated_tokens": estimated_tokens,
        "estimated_cost_usd": estimated_cost,
        "task_completed": task_completed,
    }


def _persist_eval_run(session_id: str, state: dict, forced_mode: str, elapsed_sec: float) -> None:
    metrics = _compute_metrics(state, elapsed_sec)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO eval_runs
                (session_id, architecture_mode, execution_time_sec, llm_call_count,
                 estimated_tokens, estimated_cost_usd, task_completed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                forced_mode,
                metrics["execution_time_sec"],
                metrics["llm_call_count"],
                metrics["estimated_tokens"],
                metrics["estimated_cost_usd"],
                int(metrics["task_completed"]),
            ),
        )
        conn.commit()


def _persist_session(session_id: str, state: dict, forced_mode: str | None, elapsed_sec: float) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions
                (session_id, user_query, dataset_names, complexity_score,
                 complexity_level, routing_reason, forced_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                state.get("user_query", ""),
                json.dumps(state.get("dataset_names", [])),
                state.get("complexity_score"),
                state.get("complexity_level"),
                state.get("routing_reason"),
                forced_mode,
            ),
        )
        conn.commit()
