"""
Execution trace logging helper.

Module: M9 (Execution Trace & Logging)

Every agent node calls `log_trace()` after completing a step. This does two things:
  1. Appends a structured entry to the in-memory `state["execution_trace"]` list
     (so downstream agents / the final report can reference prior decisions).
  2. Persists the same entry to the `execution_trace` SQLite table for the
     Streamlit "Execution Trace" tab and for post-hoc analysis.

This is the single mechanism that makes the system's decisions explainable
(PRD Section 14 / Section 6 agent contracts).
"""
import time
from datetime import datetime, timezone
from typing import Any, Optional

from app.db.connection import get_connection


def log_trace(
    state: dict,
    session_id: str,
    agent_name: str,
    action: str,
    reason: str,
    tool_used: Optional[str] = None,
    llm_call: bool = False,
    duration_ms: Optional[int] = None,
) -> dict:
    """
    Record one agent decision/action.

    Args:
        state: the shared AgentState dict (mutated in place; also returned for convenience)
        session_id: the session this trace entry belongs to
        agent_name: e.g. "orchestrator", "data_prep_agent", "analysis_agent",
                    "insight_validation_agent"
        action: short machine-friendly action name, e.g. "complexity_assessment",
                "impute_missing", "generate_insight", "validate_claim"
        reason: human-readable justification -- this is what gets shown in the UI
        tool_used: name of the deterministic tool/function invoked, if any
        llm_call: whether this step involved an LLM call (used for cost/efficiency metrics)
        duration_ms: how long the step took, if measured by the caller

    Returns:
        The same `state` dict, with the new entry appended to state["execution_trace"].
    """
    entry = {
        "session_id": session_id,
        "step_order": len(state.get("execution_trace", [])) + 1,
        "agent_name": agent_name,
        "action": action,
        "reason": reason,
        "tool_used": tool_used,
        "llm_call": llm_call,
        "duration_ms": duration_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    state.setdefault("execution_trace", []).append(entry)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO execution_trace
                (session_id, step_order, agent_name, action, reason, tool_used, llm_call, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["session_id"],
                entry["step_order"],
                entry["agent_name"],
                entry["action"],
                entry["reason"],
                entry["tool_used"],
                int(entry["llm_call"]),
                entry["duration_ms"],
            ),
        )
        conn.commit()

    return state


class StepTimer:
    """Small convenience context manager for measuring duration_ms around a step."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.duration_ms = int((time.perf_counter() - self._start) * 1000)


def get_trace_for_session(session_id: str) -> list[dict[str, Any]]:
    """Fetch the full ordered execution trace for a session (used by the /trace API + UI)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT step_order, agent_name, action, reason, tool_used, llm_call, duration_ms, timestamp
            FROM execution_trace
            WHERE session_id = ?
            ORDER BY step_order ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]
