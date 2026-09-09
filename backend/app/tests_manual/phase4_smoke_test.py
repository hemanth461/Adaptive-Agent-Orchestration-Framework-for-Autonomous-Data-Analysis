"""
Manual smoke test for Phase 4: Report Generation.

Reuses the Phase 2 (simple) and Phase 3 (complex) scenarios but now checks
that state["final_report"] is populated with a well-formed markdown report
in both cases. LLM calls mocked for offline testing.

Run with:
    python -m app.tests_manual.phase4_smoke_test
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
from app.core import report_generator
from app.agents import data_prep_agent
from app.agents import insight_validation_agent


def _mock_parse_simple(user_query: str) -> dict:
    return {"operations": ["descriptive_stats", "bar_chart"], "summary": user_query[:100]}


def _mock_parse_complex(user_query: str) -> dict:
    return {
        "operations": [
            "clean_missing_values", "clean_duplicates", "merge_datasets",
            "correlation_analysis", "group_comparison", "trend_analysis",
            "bar_chart", "heatmap", "statistical_validation", "multiple_objectives",
        ],
        "summary": user_query[:100],
    }


def _mock_tiebreaker(user_query, task_operations, profile_summary, score, level_a, level_b):
    return {"level": level_b, "reason": "Mocked tiebreaker."}


def _mock_imputation(df, numeric_cols, categorical_cols):
    return {col: "median" for col in numeric_cols}


def _mock_insights(analysis_results, user_query):
    gc = analysis_results.get("group_comparison", {}).get("result", {})
    groups = list(gc.items())
    if len(groups) < 2:
        return []
    (g1, s1), (g2, s2) = groups[0], groups[1]
    return [{
        "claim": f"Group '{g1}' differs from group '{g2}' in average spend",
        "evidence": {
            f"{g1}_mean": s1["mean"], f"{g1}_n": s1["count"],
            f"{g2}_mean": s2["mean"], f"{g2}_n": s2["count"],
        },
        "claim_type": "group_comparison",
    }]


def _mock_exec_summary(state):
    return (
        f"This {state.get('complexity_level')} analysis processed the requested "
        f"operations and produced {len(state.get('visualizations', []))} visualization(s) "
        f"and {len(state.get('insights', []))} insight(s)."
    )


complexity_scoring._llm_tiebreaker = _mock_tiebreaker
data_prep_agent._llm_choose_imputation_strategies = _mock_imputation
insight_validation_agent._generate_insights = _mock_insights
report_generator._generate_executive_summary = _mock_exec_summary

from app.core.graph import get_graph  # noqa: E402
from app.agents import orchestrator as orchestrator_module  # noqa: E402


def make_sales_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "region": rng.choice(["North", "South", "East"], size=60),
        "sales": rng.normal(4800, 700, size=60).round(2),
    })


def make_customer_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n_north, n_south = 90, 60
    df = pd.DataFrame({
        "customer_id": range(1, n_north + n_south + 1),
        "region": ["North"] * n_north + ["South"] * n_south,
        "spend": np.concatenate([
            rng.normal(260, 40, size=n_north),
            rng.normal(180, 40, size=n_south),
        ]).round(2),
        "signup_score": rng.normal(70, 15, size=n_north + n_south).round(1),
    })
    df.loc[df.sample(10, random_state=1).index, "spend"] = np.nan
    return pd.concat([df, df.iloc[:8]], ignore_index=True)


def run_case(name: str, df: pd.DataFrame, query: str, mock_parser):
    orchestrator_module.parse_user_query = mock_parser
    session_id = str(uuid.uuid4())

    initial_state = {
        "session_id": session_id,
        "raw_datasets": [df],
        "dataset_names": ["dataset"],
        "user_query": query,
        "execution_trace": [],
        "retry_count": 0,
    }

    graph = get_graph()
    result = graph.invoke(initial_state)

    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"Level: {result['complexity_level']} | Score: {result['complexity_score']}")
    report = result.get("final_report", "")
    print(f"Report generated: {bool(report)} | length: {len(report)} chars")
    print("\n--- REPORT PREVIEW ---")
    print(report[:1200])
    print("..." if len(report) > 1200 else "")
    return result


if __name__ == "__main__":
    init_db()

    run_case(
        "Case A: SIMPLE path report",
        make_sales_dataset(),
        "Calculate the average sales and create a bar chart by region.",
        _mock_parse_simple,
    )

    run_case(
        "Case B: COMPLEX path report",
        make_customer_dataset(),
        "Clean the data, compare spend by region, show correlation, and validate significance.",
        _mock_parse_complex,
    )
