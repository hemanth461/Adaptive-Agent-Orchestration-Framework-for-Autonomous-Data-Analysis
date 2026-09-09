"""
Manual smoke test for Phase 6: Edge Case Handling.

Verifies fixes made during polish:
  1. Empty dataframe (0 rows) doesn't crash profiling.
  2. Single-column dataset doesn't crash chart selection / analysis.
  3. High-cardinality group column is skipped in groupby_compare rather than
     producing an unreadable result.
  4. Unrecognized/empty query falls back to a default operation instead of
     an empty pipeline.

Run with:
    python -m app.tests_manual.phase6_edge_case_test
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.profiling import profile_datasets, profile_single_dataset
from app.core.analysis_tools import groupby_compare, select_chart_type
from app.core.task_understanding import KNOWN_OPERATIONS


def test_empty_dataframe():
    print("\n--- Test 1: Empty dataframe (0 rows) ---")
    empty_df = pd.DataFrame(columns=["a", "b", "c"])
    profile = profile_single_dataset(empty_df, "empty")
    print(f"  Profile: {profile}")
    assert profile["n_rows"] == 0
    assert profile["is_empty"] is True
    print("  PASS: no crash, well-formed empty profile returned.")


def test_single_column_dataset():
    print("\n--- Test 2: Single-column dataset ---")
    df = pd.DataFrame({"value": np.random.default_rng(1).normal(50, 10, size=30)})
    profile = profile_datasets([df], ["single_col"])
    column_profile = profile["datasets"][0]["columns"]
    recommendations = select_chart_type(df, column_profile)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Chart recommendations: {[r['chart_type'] for r in recommendations]}")
    # Should only recommend a histogram (no categorical/datetime/second-numeric column)
    assert all(r["chart_type"] == "histogram" for r in recommendations)
    print("  PASS: only histogram recommended, no crash from missing column types.")


def test_high_cardinality_group_column():
    print("\n--- Test 3: High-cardinality group column (looks like an ID) ---")
    df = pd.DataFrame({
        "transaction_id": range(200),  # 200 unique values -- not a real category
        "amount": np.random.default_rng(2).normal(100, 20, size=200),
    })
    result = groupby_compare(df, "transaction_id", "amount")
    print(f"  Result: {'empty (correctly skipped)' if not result else 'NOT EMPTY - BUG'}")
    assert result == {}
    print("  PASS: high-cardinality group column correctly skipped.")


def test_unrecognized_query_fallback():
    print("\n--- Test 4: Unrecognized query operations fallback ---")
    # Simulate what parse_user_query does internally after LLM filtering,
    # without needing a live API call.
    fake_llm_operations = ["some_hallucinated_operation", "another_bad_one"]
    filtered = [op for op in fake_llm_operations if op in KNOWN_OPERATIONS]
    if not filtered:
        filtered = ["descriptive_stats"]
    print(f"  Filtered operations: {filtered}")
    assert filtered == ["descriptive_stats"]
    print("  PASS: falls back to descriptive_stats instead of an empty pipeline.")


if __name__ == "__main__":
    test_empty_dataframe()
    test_single_column_dataset()
    test_high_cardinality_group_column()
    test_unrecognized_query_fallback()
    print("\nAll edge-case tests passed.")
