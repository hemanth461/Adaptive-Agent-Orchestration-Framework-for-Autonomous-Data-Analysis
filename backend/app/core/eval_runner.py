"""
Evaluation Harness Runner.

Module: M13 (Evaluation Harness)

Runs each case in EVAL_CASES through three modes:
    - "single"    : forced simple/single-agent path regardless of true complexity
    - "multi"     : forced complex/multi-agent path regardless of true complexity
    - "adaptive"  : normal adaptive routing (the system's actual behavior)

Produces a comparison table per PRD Section 13: execution time, LLM call
count, estimated cost, task completion, and whether adaptive's choice
matched the case's human-judged `expected_level`.

Run with:
    python -m app.core.eval_runner
"""
import copy
import pandas as pd

from app.core.eval_cases import EVAL_CASES
from app.core.session_runner import run_session
from app.db.connection import get_connection, init_db

MODES = ["single", "multi", "adaptive"]


def run_all_cases(cases: list[dict] | None = None) -> pd.DataFrame:
    """Run every case through every mode; returns a tidy results DataFrame."""
    init_db()
    cases = cases if cases is not None else EVAL_CASES
    rows = []

    for case in cases:
        for mode in MODES:
            # dataframes must be re-copied per run since agents mutate in place
            dataframes = [df.copy() for df in case["dataframes"]]

            print(f"Running case='{case['case_id']}' mode='{mode}'...")
            result = run_session(
                dataframes=dataframes,
                dataset_names=case["dataset_names"],
                user_query=case["query"],
                forced_mode=mode,
            )

            trace = result.get("execution_trace", [])
            llm_calls = sum(1 for t in trace if t.get("llm_call"))
            adaptive_choice = result.get("adaptive_recommended_level", result.get("complexity_level"))

            rows.append({
                "case_id": case["case_id"],
                "description": case["description"],
                "mode": mode,
                "expected_level": case["expected_level"],
                "adaptive_recommended_level": adaptive_choice,
                "actual_level_used": result.get("complexity_level"),
                "complexity_score": result.get("complexity_score"),
                "n_trace_steps": len(trace),
                "llm_call_count": llm_calls,
                "estimated_tokens": llm_calls * 600,
                "estimated_cost_usd": round((llm_calls * 600 / 1000) * 0.003, 5),
                "n_visualizations": len(result.get("visualizations", [])),
                "n_insights": len(result.get("insights", [])),
                "retry_count": result.get("retry_count", 0),
                "task_completed": bool(result.get("final_report")),
                "adaptive_matched_expected": (
                    adaptive_choice == case["expected_level"] if mode == "adaptive" else None
                ),
            })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate comparison across modes: PRD Section 13 metrics table."""
    summary = df.groupby("mode").agg(
        avg_llm_calls=("llm_call_count", "mean"),
        avg_estimated_cost_usd=("estimated_cost_usd", "mean"),
        avg_trace_steps=("n_trace_steps", "mean"),
        completion_rate=("task_completed", "mean"),
    ).round(4)
    return summary


def adaptive_accuracy(df: pd.DataFrame) -> dict:
    """What fraction of cases did adaptive routing match the expected human-judged level?"""
    adaptive_rows = df[df["mode"] == "adaptive"]
    matched = adaptive_rows["adaptive_matched_expected"].sum()
    total = len(adaptive_rows)
    return {
        "matched": int(matched),
        "total": total,
        "accuracy": round(matched / total, 3) if total else None,
        "mismatches": adaptive_rows[adaptive_rows["adaptive_matched_expected"] == False][  # noqa: E712
            ["case_id", "expected_level", "adaptive_recommended_level"]
        ].to_dict(orient="records"),
    }


if __name__ == "__main__":
    results_df = run_all_cases()

    print("\n" + "=" * 80)
    print("FULL RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (avg across cases, by mode)")
    print("=" * 80)
    print(summarize(results_df).to_string())

    print("\n" + "=" * 80)
    print("ADAPTIVE ROUTING ACCURACY (vs. human-judged expected level)")
    print("=" * 80)
    acc = adaptive_accuracy(results_df)
    print(f"Matched {acc['matched']}/{acc['total']} cases ({acc['accuracy']})")
    if acc["mismatches"]:
        print("Mismatches:")
        for m in acc["mismatches"]:
            print(f"  - {m['case_id']}: expected={m['expected_level']}, "
                  f"adaptive_chose={m['adaptive_recommended_level']}")

    results_df.to_csv("eval_results.csv", index=False)
    print("\nSaved full results to eval_results.csv")
