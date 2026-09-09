"""
Manual smoke test for Phase 2: Orchestrator + General Agent path via LangGraph.

LLM calls (task_understanding.parse_user_query and the complexity tiebreaker)
are monkeypatched with deterministic stand-ins so this runs without an API key.
Once ANTHROPIC_API_KEY is set, delete the monkeypatch block to test for real.

Run with:
    python -m app.tests_manual.phase2_smoke_test
"""
import sys
import os
import uuid
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.db.connection import init_db
from app.core import task_understanding
from app.core import complexity_scoring


def _mock_parse_user_query(user_query: str) -> dict:
    """Deterministic stand-in for the LLM query parser."""
    ops = []
    q = user_query.lower()
    if "average" in q or "mean" in q or "stats" in q:
        ops.append("descriptive_stats")
    if "bar chart" in q or "bar" in q:
        ops.append("bar_chart")
    if "correlation" in q:
        ops.append("correlation_analysis")
    if "heatmap" in q:
        ops.append("heatmap")
    return {"operations": ops, "summary": user_query[:100]}


def _mock_llm_tiebreaker(user_query, task_operations, profile_summary, score, level_a, level_b):
    return {"level": level_a, "reason": "Mocked tiebreaker: defaulted to lower complexity."}


# Monkeypatch before importing modules that reference these at call-time
task_understanding.parse_user_query = _mock_parse_user_query
complexity_scoring._llm_tiebreaker = _mock_llm_tiebreaker

from app.core.graph import get_graph  # noqa: E402  (import after monkeypatch)


def make_sales_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "region": rng.choice(["North", "South", "East"], size=60),
        "sales": rng.normal(4800, 700, size=60).round(2),
    })


def run_simple_case():
    init_db()
    session_id = str(uuid.uuid4())
    df = make_sales_dataset()

    initial_state = {
        "session_id": session_id,
        "raw_datasets": [df],
        "dataset_names": ["sales"],
        "user_query": "Calculate the average sales and create a bar chart by region.",
        "execution_trace": [],
        "retry_count": 0,
    }

    graph = get_graph()
    result = graph.invoke(initial_state)

    print(f"\n=== Simple Case Result (session {session_id[:8]}) ===")
    print(f"Complexity: {result['complexity_score']} -> {result['complexity_level'].upper()}")
    print(f"Routing reason: {result['routing_reason']}")
    print(f"Task operations: {result['task_operations']}")
    print(f"Analysis results keys: {list(result['analysis_results'].keys())}")
    print(f"Visualizations generated: {len(result['visualizations'])}")
    for v in result["visualizations"]:
        print(f"  - {v['type']} ({v['columns']}) -> {v['path']}")
        print(f"    reason: {v['reason']}")
    print(f"\nExecution trace ({len(result['execution_trace'])} steps):")
    for step in result["execution_trace"]:
        print(f"  [{step['step_order']}] {step['agent_name']} :: {step['action']}")
        print(f"       -> {step['reason']}")


if __name__ == "__main__":
    run_simple_case()
