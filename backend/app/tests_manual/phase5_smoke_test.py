"""
Manual smoke test for Phase 5: Evaluation Harness.

Uses a GENERIC (not per-case-hardcoded) keyword-based mock for the query
parser, so this exercises the harness mechanics honestly across all 6 fixed
cases x 3 modes = 18 runs. Once ANTHROPIC_API_KEY is set, run
`python -m app.core.eval_runner` directly instead -- this script is only
for offline verification that forced_mode routing and metrics computation work.

Run with:
    python -m app.tests_manual.phase5_smoke_test
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core import complexity_scoring
from app.agents import data_prep_agent
from app.agents import insight_validation_agent
from app.core import report_generator

KEYWORD_MAP = {
    "clean": ["clean_missing_values"],
    "duplicate": ["clean_duplicates"],
    "correlation": ["correlation_analysis"],
    "compare": ["group_comparison"],
    "trend": ["trend_analysis"],
    "bar chart": ["bar_chart"],
    "scatter": ["scatter_plot"],
    "heatmap": ["heatmap"],
    "visualiz": ["bar_chart", "heatmap"],
    "descriptive statistics": ["descriptive_stats"],
    "average": ["descriptive_stats"],
    "validat": ["statistical_validation"],
    "significant": ["statistical_validation"],
}


def _mock_parse_user_query(user_query: str) -> dict:
    q = user_query.lower()
    ops = set()
    for kw, mapped in KEYWORD_MAP.items():
        if kw in q:
            ops.update(mapped)
    if len(ops) >= 4:
        ops.add("multiple_objectives")
    return {"operations": sorted(ops), "summary": user_query[:100]}


def _mock_tiebreaker(user_query, task_operations, profile_summary, score, level_a, level_b):
    # Generic heuristic: pick the higher-complexity option if validation was
    # requested, otherwise the lower one -- mirrors a reasonable real judgment
    # without hardcoding per-case answers.
    level = level_b if "statistical_validation" in task_operations else level_a
    return {"level": level, "reason": "Mocked tiebreaker (generic heuristic)."}


def _mock_imputation(df, numeric_cols, categorical_cols):
    return {col: ("median" if abs(df[col].skew()) > 0.5 else "mean") for col in numeric_cols}


def _mock_insights(analysis_results, user_query):
    insights = []
    gc = analysis_results.get("group_comparison", {})
    if gc:
        result = gc.get("result", {})
        groups = list(result.items())
        if len(groups) >= 2:
            (g1, s1), (g2, s2) = groups[0], groups[1]
            insights.append({
                "claim": f"Group '{g1}' differs from group '{g2}' in {gc.get('value_col', 'value')}",
                "evidence": {
                    f"{g1}_mean": s1.get("mean"), f"{g1}_n": s1.get("count"),
                    f"{g2}_mean": s2.get("mean"), f"{g2}_n": s2.get("count"),
                },
                "claim_type": "group_comparison",
            })
    return insights


def _mock_exec_summary(state):
    return (
        f"This {state.get('complexity_level')} analysis produced "
        f"{len(state.get('visualizations', []))} visualization(s) and "
        f"{len(state.get('insights', []))} insight(s)."
    )


complexity_scoring._llm_tiebreaker = _mock_tiebreaker
data_prep_agent._llm_choose_imputation_strategies = _mock_imputation
insight_validation_agent._generate_insights = _mock_insights
report_generator._generate_executive_summary = _mock_exec_summary

from app.agents import orchestrator as orchestrator_module  # noqa: E402
orchestrator_module.parse_user_query = _mock_parse_user_query

from app.core.eval_runner import run_all_cases, summarize, adaptive_accuracy  # noqa: E402


if __name__ == "__main__":
    results_df = run_all_cases()

    print("\n" + "=" * 100)
    print("FULL RESULTS")
    print("=" * 100)
    cols = ["case_id", "mode", "expected_level", "actual_level_used",
            "complexity_score", "llm_call_count", "task_completed", "retry_count"]
    print(results_df[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY (avg across cases, by mode)")
    print("=" * 100)
    print(summarize(results_df).to_string())

    print("\n" + "=" * 100)
    print("ADAPTIVE ROUTING ACCURACY (vs. human-judged expected level)")
    print("=" * 100)
    acc = adaptive_accuracy(results_df)
    print(f"Matched {acc['matched']}/{acc['total']} cases ({acc['accuracy']})")
    for m in acc["mismatches"]:
        print(f"  - MISMATCH: {m['case_id']}: expected={m['expected_level']}, "
              f"adaptive_chose={m['adaptive_recommended_level']}")
