"""
Analysis & Visualization Tools.

Module: M5 (General Agent tools) / reused by M7 (Analysis Agent, Phase 3)

Pure, deterministic functions -- no LLM calls. Charts are saved as PNG files
under data/outputs/{session_id}/ and referenced by path in the returned dict,
so both FastAPI and Streamlit can serve them without re-generating anything.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "outputs")


def _ensure_output_dir(session_id: str) -> str:
    path = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Statistics tools
# ---------------------------------------------------------------------------

def describe_stats(df: pd.DataFrame) -> dict:
    """Descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}
    return {
        col: {
            "mean": round(float(numeric_df[col].mean()), 3),
            "median": round(float(numeric_df[col].median()), 3),
            "std": round(float(numeric_df[col].std()), 3),
            "min": round(float(numeric_df[col].min()), 3),
            "max": round(float(numeric_df[col].max()), 3),
        }
        for col in numeric_df.columns
    }


def correlation_matrix(df: pd.DataFrame) -> dict:
    """
    Pearson correlation matrix for numeric columns, as a nested dict.

    Excludes ID-like columns (e.g. `customer_id`) from the numeric set --
    correlating an identifier with anything is statistically meaningless and
    misleads downstream insight generation.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    id_like = [c for c in numeric_df.columns if c.lower().endswith("_id") or c.lower() == "id"]
    numeric_df = numeric_df.drop(columns=id_like)
    if numeric_df.shape[1] < 2:
        return {}
    corr = numeric_df.corr().round(3)
    return corr.to_dict()


def groupby_compare(df: pd.DataFrame, group_col: str, value_col: str, max_groups: int = 20) -> dict:
    """
    Compare a numeric value column across groups defined by a categorical column.

    Edge case: if group_col has more than `max_groups` unique values, this is
    likely a high-cardinality/ID-like column rather than a genuine category --
    return an empty result rather than producing an unreadable 500-row table.
    """
    if group_col not in df.columns or value_col not in df.columns:
        return {}
    if df[group_col].nunique(dropna=True) > max_groups:
        return {}
    grouped = df.groupby(group_col)[value_col].agg(["mean", "count", "std"]).round(3)
    return grouped.to_dict(orient="index")


# ---------------------------------------------------------------------------
# Visualization tools -- each returns {"type", "path", "columns"}
# ---------------------------------------------------------------------------

def plot_bar(df: pd.DataFrame, category_col: str, value_col: str, session_id: str) -> dict:
    out_dir = _ensure_output_dir(session_id)
    grouped = df.groupby(category_col)[value_col].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(f"Average {value_col} by {category_col}")
    ax.set_ylabel(value_col)
    fig.tight_layout()
    path = os.path.join(out_dir, f"bar_{category_col}_{value_col}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"type": "bar_chart", "path": path, "columns": [category_col, value_col]}


def plot_line(df: pd.DataFrame, time_col: str, value_col: str, session_id: str) -> dict:
    out_dir = _ensure_output_dir(session_id)
    plot_df = df[[time_col, value_col]].dropna().copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
    plot_df = plot_df.sort_values(time_col)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(plot_df[time_col], plot_df[value_col], color="#55A868")
    ax.set_title(f"{value_col} over time")
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(out_dir, f"line_{time_col}_{value_col}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"type": "line_chart", "path": path, "columns": [time_col, value_col]}


def plot_histogram(df: pd.DataFrame, col: str, session_id: str) -> dict:
    out_dir = _ensure_output_dir(session_id)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df[col].dropna().plot(kind="hist", bins=25, ax=ax, color="#C44E52")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    fig.tight_layout()
    path = os.path.join(out_dir, f"hist_{col}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"type": "histogram", "path": path, "columns": [col]}


def plot_scatter(df: pd.DataFrame, col_x: str, col_y: str, session_id: str) -> dict:
    out_dir = _ensure_output_dir(session_id)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df[col_x], df[col_y], alpha=0.6, color="#8172B2", s=18)
    ax.set_title(f"{col_y} vs {col_x}")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    fig.tight_layout()
    path = os.path.join(out_dir, f"scatter_{col_x}_{col_y}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"type": "scatter_plot", "path": path, "columns": [col_x, col_y]}


def plot_heatmap(df: pd.DataFrame, session_id: str) -> dict:
    out_dir = _ensure_output_dir(session_id)
    numeric_df = df.select_dtypes(include=[np.number])
    id_like = [c for c in numeric_df.columns if c.lower().endswith("_id") or c.lower() == "id"]
    numeric_df = numeric_df.drop(columns=id_like)
    if numeric_df.shape[1] < 2:
        return {}
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    path = os.path.join(out_dir, "heatmap.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"type": "heatmap", "path": path, "columns": list(corr.columns)}


# ---------------------------------------------------------------------------
# Rule-based chart-type selection (PRD Section 6, Agent 2)
# ---------------------------------------------------------------------------

def select_chart_type(df: pd.DataFrame, column_profile: dict) -> list[dict]:
    """
    Deterministic chart-type selection rules.

    Returns a list of {"chart_type", "columns", "rule_reason"} recommendations
    based on column dtypes, to be executed by the caller (General Agent or
    Analysis Agent) and then justified via a single LLM call over these
    already-decided rule outcomes (not a free re-decision).
    """
    recommendations = []
    numeric_cols = [
        c for c, meta in column_profile.items()
        if meta["is_numeric"] and not (c.lower().endswith("_id") or c.lower() == "id")
    ]
    datetime_cols = [c for c, meta in column_profile.items() if meta["is_datetime"]]
    categorical_cols = [
        c for c, meta in column_profile.items()
        if not meta["is_numeric"] and not meta["is_datetime"] and meta["n_unique"] <= 20
    ]

    if datetime_cols and numeric_cols:
        recommendations.append({
            "chart_type": "line_chart",
            "columns": [datetime_cols[0], numeric_cols[0]],
            "rule_reason": "Dataset contains a time-based column paired with a numeric "
                            "column, indicating a trend-over-time relationship.",
        })

    if categorical_cols and numeric_cols:
        recommendations.append({
            "chart_type": "bar_chart",
            "columns": [categorical_cols[0], numeric_cols[0]],
            "rule_reason": "A categorical column and a numeric column suggest a "
                            "category comparison, best shown as a bar chart.",
        })

    if len(numeric_cols) >= 2:
        recommendations.append({
            "chart_type": "scatter_plot",
            "columns": [numeric_cols[0], numeric_cols[1]],
            "rule_reason": "Two numeric columns are present, suggesting a possible "
                            "relationship best visualized with a scatter plot.",
        })
        recommendations.append({
            "chart_type": "heatmap",
            "columns": numeric_cols,
            "rule_reason": "Multiple numeric columns allow a correlation heatmap "
                            "to summarize pairwise relationships at a glance.",
        })

    if numeric_cols:
        recommendations.append({
            "chart_type": "histogram",
            "columns": [numeric_cols[0]],
            "rule_reason": "A numeric column's distribution shape is best shown "
                            "with a histogram.",
        })

    return recommendations
