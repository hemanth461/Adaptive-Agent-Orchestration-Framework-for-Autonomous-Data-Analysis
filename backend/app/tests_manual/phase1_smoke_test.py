"""
Manual smoke test for Phase 1: Profiling + Complexity Scoring.

Runs WITHOUT calling any LLM (bypasses the tiebreaker path) so it can be
verified immediately, before wiring up an API key.

Run with:
    python -m app.tests_manual.phase1_smoke_test
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.profiling import profile_datasets
from app.core.complexity_scoring import compute_complexity_score, THRESHOLDS


def make_clean_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "region": rng.choice(["A", "B", "C"], size=100),
        "sales": rng.normal(5000, 800, size=100).round(2),
    })


def make_messy_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "signup_date": pd.date_range("2023-01-01", periods=200, freq="3D").astype(str),
        "spend": rng.normal(200, 60, size=200).round(2),
        "country": rng.choice(["US", "IN", "UK", None], size=200, p=[0.4, 0.3, 0.2, 0.1]),
        "email": [f"user{i}@mail.com" if i % 15 != 0 else None for i in range(200)],
    })
    # inject duplicates
    df = pd.concat([df, df.iloc[:15]], ignore_index=True)
    return df


def run_case(name: str, dataframes, dataset_names, task_operations):
    profile = profile_datasets(dataframes, dataset_names)
    scored = compute_complexity_score(profile, task_operations)

    print(f"\n=== Case: {name} ===")
    print(f"Datasets: {dataset_names}")
    print(f"Operations: {task_operations}")
    print(f"Profile summary: n_datasets={profile['n_datasets']}, "
          f"n_columns={profile['n_columns']}, missing_pct={profile['missing_pct']}, "
          f"duplicate_pct={profile['duplicate_pct']}")
    print(f"Score: {scored['total_score']} / 10")
    print(f"Breakdown: {scored['breakdown']}")

    score = scored["total_score"]
    if score <= THRESHOLDS["simple_max"]:
        level = "SIMPLE"
    elif score <= THRESHOLDS["moderate_ambiguous_high"]:
        level = "SIMPLE/MODERATE (ambiguous band -> would trigger LLM tiebreaker)"
    elif score <= THRESHOLDS["complex_ambiguous_low"]:
        level = "MODERATE"
    elif score <= THRESHOLDS["complex_ambiguous_high"]:
        level = "MODERATE/COMPLEX (ambiguous band -> would trigger LLM tiebreaker)"
    else:
        level = "COMPLEX"
    print(f"Predicted level: {level}")


if __name__ == "__main__":
    # Case 1: Simple -- one clean dataset, one basic request
    run_case(
        "Simple - clean data, basic stats",
        [make_clean_dataset()],
        ["sales_clean"],
        ["descriptive_stats", "bar_chart"],
    )

    # Case 2: Moderate -- one dataset with minor issues, a few operations
    run_case(
        "Moderate - minor cleaning + correlation + charts",
        [make_messy_dataset()],
        ["customers"],
        ["clean_missing_values", "correlation_analysis", "bar_chart", "histogram"],
    )

    # Case 3: Complex -- multiple datasets, messy, multi-objective + validation
    run_case(
        "Complex - multi-dataset, messy, validation required",
        [make_messy_dataset(), make_clean_dataset()],
        ["customers", "sales"],
        [
            "clean_missing_values",
            "clean_duplicates",
            "merge_datasets",
            "trend_analysis",
            "group_comparison",
            "correlation_analysis",
            "bar_chart",
            "line_chart",
            "heatmap",
            "statistical_validation",
            "multiple_objectives",
        ],
    )
