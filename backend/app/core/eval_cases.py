"""
Fixed Evaluation Test Cases.

Module: M13 (Evaluation Harness)

A fixed set of dataset + query pairs, chosen to cover:
  - clean + simple
  - minor issues + moderate
  - messy + multi-objective + complex
  - boundary/ambiguous cases near the scoring thresholds

Kept small (6 cases) per the PRD's realistic-scope guidance (Section 12/13:
"5-10 cases... framed as demonstration, not research"). Each case also
carries `expected_level` -- the level a human would reasonably pick -- used
to check whether adaptive routing's choice matches human judgment.
"""
import numpy as np
import pandas as pd


def _clean_sales() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "region": rng.choice(["North", "South", "East"], size=60),
        "sales": rng.normal(4800, 700, size=60).round(2),
    })


def _minor_issues_marketing() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "campaign": rng.choice(["A", "B", "C"], size=120),
        "impressions": rng.integers(1000, 5000, size=120),
        "clicks": rng.integers(10, 400, size=120),
        "spend": rng.normal(150, 30, size=120).round(2),
    })
    df.loc[df.sample(6, random_state=2).index, "spend"] = np.nan
    return df


def _messy_customers() -> pd.DataFrame:
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


def _messy_orders() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "order_id": range(1, 181),
        "category": rng.choice(["Electronics", "Home", "Toys", None], size=180, p=[0.35, 0.3, 0.25, 0.1]),
        "amount": rng.normal(80, 25, size=180).round(2),
        "order_date": pd.date_range("2023-06-01", periods=180, freq="2D").astype(str),
    })
    return pd.concat([df, df.iloc[:12]], ignore_index=True)


def _boundary_low() -> pd.DataFrame:
    """Designed to score near the Simple/Moderate boundary (~3.5-4.5)."""
    rng = np.random.default_rng(9)
    return pd.DataFrame({
        "product": rng.choice(["X", "Y"], size=80),
        "price": rng.normal(50, 10, size=80).round(2),
        "units_sold": rng.integers(1, 50, size=80),
    })


def _boundary_high() -> pd.DataFrame:
    """Designed to score near the Moderate/Complex boundary (~6.5-8.0)."""
    rng = np.random.default_rng(13)
    df = pd.DataFrame({
        "employee_id": range(1, 101),
        "department": rng.choice(["Eng", "Sales", "Ops"], size=100),
        "tenure_years": rng.integers(0, 15, size=100),
        "salary": rng.normal(75000, 15000, size=100).round(2),
    })
    df.loc[df.sample(8, random_state=4).index, "salary"] = np.nan
    return df


EVAL_CASES = [
    {
        "case_id": "simple_clean_sales",
        "description": "Clean dataset, minimal request",
        "dataframes": [_clean_sales()],
        "dataset_names": ["sales"],
        "query": "Calculate the average sales and create a bar chart by region.",
        "expected_level": "simple",
    },
    {
        "case_id": "moderate_marketing",
        "description": "Minor missing data, a few operations",
        "dataframes": [_minor_issues_marketing()],
        "dataset_names": ["marketing"],
        "query": "Clean minor missing values, analyze correlations between "
                  "impressions, clicks, and spend, and create a heatmap.",
        "expected_level": "moderate",
    },
    {
        "case_id": "complex_customers",
        "description": "Messy data, multiple objectives, validation required",
        "dataframes": [_messy_customers()],
        "dataset_names": ["customers"],
        "query": "Clean the data, compare spend by region, show correlation "
                  "with signup score, generate visualizations, and validate "
                  "whether the regional difference is statistically significant.",
        "expected_level": "complex",
    },
    {
        "case_id": "complex_orders",
        "description": "Messy multi-column categorical data, trend + validation",
        "dataframes": [_messy_orders()],
        "dataset_names": ["orders"],
        "query": "Clean duplicates and missing categories, analyze order amount "
                  "trends over time by category, visualize with multiple charts, "
                  "and validate the key findings statistically.",
        "expected_level": "complex",
    },
    {
        "case_id": "boundary_simple_moderate",
        "description": "Near the Simple/Moderate scoring threshold",
        "dataframes": [_boundary_low()],
        "dataset_names": ["products"],
        "query": "Show descriptive statistics and a scatter plot of price vs units sold.",
        "expected_level": "simple",
    },
    {
        "case_id": "boundary_moderate_complex",
        "description": "Near the Moderate/Complex scoring threshold",
        "dataframes": [_boundary_high()],
        "dataset_names": ["employees"],
        "query": "Clean missing salary values, compare salary by department, "
                  "check correlation with tenure, and validate if the "
                  "department difference is significant.",
        "expected_level": "moderate",
    },
]
