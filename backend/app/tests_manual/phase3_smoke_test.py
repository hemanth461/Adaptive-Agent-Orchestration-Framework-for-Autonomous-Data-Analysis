"""
Manual smoke test for Phase 3: Complex path (Data Prep -> Analysis ->
Insight/Validation) with the bounded retry loop.

LLM call sites are monkeypatched with deterministic stand-ins so this runs
without an API key. Two scenarios are tested:
  1. A claim that validates as supported on the first pass (no retry).
  2. A claim that is deliberately unsupported (low sample size), to confirm
     the retry fires exactly once and then terminates instead of looping.

Run with:
    python -m app.tests_manual.phase3_smoke_test
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
from app.agents import data_prep_agent
from app.agents import insight_validation_agent


# --- Monkeypatch all LLM call sites for offline testing ---

def _mock_parse_user_query(user_query: str) -> dict:
    return {
        "operations": [
            "clean_missing_values", "clean_duplicates", "merge_datasets",
            "correlation_analysis", "group_comparison", "trend_analysis",
            "bar_chart", "heatmap", "statistical_validation", "multiple_objectives",
        ],
        "summary": user_query[:100],
    }


def _mock_llm_tiebreaker(user_query, task_operations, profile_summary, score, level_a, level_b):
    return {"level": level_b, "reason": "Mocked: multiple objectives + validation -> higher complexity."}


def _mock_choose_imputation_strategies(df, numeric_cols, categorical_cols):
    return {col: "median" for col in numeric_cols}


def _mock_generate_insights_supported(analysis_results, user_query):
    """Scenario 1: a well-supported claim (large sample, clear spread)."""
    gc = analysis_results.get("group_comparison", {}).get("result", {})
    if not gc:
        return []
    groups = list(gc.items())
    if len(groups) < 2:
        return []
    (g1, s1), (g2, s2) = groups[0], groups[1]
    return [{
        "claim": f"Group '{g1}' has a different average than group '{g2}'",
        "evidence": {
            f"{g1}_mean": s1["mean"], f"{g1}_n": s1["count"],
            f"{g2}_mean": s2["mean"], f"{g2}_n": s2["count"],
        },
        "claim_type": "group_comparison",
    }]


def _mock_generate_insights_unsupported(analysis_results, user_query):
    """Scenario 2: force an unsupported claim via artificially tiny sample size."""
    return [{
        "claim": "Region A has meaningfully higher sales than Region B",
        "evidence": {"region_a_mean": 5000, "region_a_n": 5, "region_b_mean": 4800, "region_b_n": 4},
        "claim_type": "group_comparison",
    }]


task_understanding.parse_user_query = _mock_parse_user_query
complexity_scoring._llm_tiebreaker = _mock_llm_tiebreaker
data_prep_agent._llm_choose_imputation_strategies = _mock_choose_imputation_strategies

from app.core.graph import get_graph  # noqa: E402


def make_messy_dataset(with_real_group_difference: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n_north, n_south = 90, 60
    north_spend_mean = 260 if with_real_group_difference else 220
    south_spend_mean = 180 if with_real_group_difference else 220

    df = pd.DataFrame({
        "customer_id": range(1, n_north + n_south + 1),
        "region": ["North"] * n_north + ["South"] * n_south,
        "spend": np.concatenate([
            rng.normal(north_spend_mean, 40, size=n_north),
            rng.normal(south_spend_mean, 40, size=n_south),
        ]).round(2),
        "signup_score": rng.normal(70, 15, size=n_north + n_south).round(1),
    })
    df.loc[df.sample(10, random_state=1).index, "spend"] = np.nan
    df = pd.concat([df, df.iloc[:8]], ignore_index=True)
    return df


def run_scenario(name: str, mock_insight_fn, session_id: str, real_group_difference: bool = False):
    insight_validation_agent._generate_insights = mock_insight_fn

    df = make_messy_dataset(with_real_group_difference=real_group_difference)
    initial_state = {
        "session_id": session_id,
        "raw_datasets": [df],
        "dataset_names": ["customers"],
        "user_query": "Clean the data, compare spend by region, show correlation, "
                       "and validate whether the regional difference is significant.",
        "execution_trace": [],
        "retry_count": 0,
    }

    graph = get_graph()
    result = graph.invoke(initial_state)

    print(f"\n{'='*70}\nScenario: {name}\n{'='*70}")
    print(f"Complexity: {result['complexity_score']} -> {result['complexity_level'].upper()}")
    print(f"Cleaning log entries: {len(result.get('cleaning_log', []))}")
    for entry in result.get("cleaning_log", [])[:5]:
        print(f"  - [{entry['issue_type']}] {entry.get('column_name')}: {entry['strategy_applied']}")
    print(f"\nInsights: {len(result.get('insights', []))}")
    for v in result.get("validation_results", []):
        print(f"  - claim: {v['claim']}")
        print(f"    supported={v['supported']} | test={v.get('test_used')} | reason={v['reason']}")
    print(f"\nFinal retry_count: {result.get('retry_count')}")
    print(f"Final retry_instruction (should be None): {result.get('retry_instruction')}")

    retry_events = [
        s for s in result["execution_trace"]
        if s["action"] in ("request_retry", "retry_group_comparison", "retry_exhausted")
    ]
    print(f"\nRetry-related trace events ({len(retry_events)}):")
    for e in retry_events:
        print(f"  [{e['agent_name']}] {e['action']}: {e['reason']}")

    print(f"\nFull execution trace ({len(result['execution_trace'])} steps):")
    for step in result["execution_trace"]:
        print(f"  [{step['step_order']}] {step['agent_name']} :: {step['action']}")


if __name__ == "__main__":
    init_db()

    run_scenario(
        "1. Claim validates as supported -> NO retry expected",
        _mock_generate_insights_supported,
        str(uuid.uuid4()),
        real_group_difference=True,
    )

    run_scenario(
        "2. Claim unsupported (tiny sample) -> ONE bounded retry, then stop",
        _mock_generate_insights_unsupported,
        str(uuid.uuid4()),
        real_group_difference=False,
    )
