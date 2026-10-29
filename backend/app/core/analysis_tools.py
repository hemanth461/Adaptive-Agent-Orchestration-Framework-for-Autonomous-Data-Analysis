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
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if 2 <= df[c].nunique(dropna=True) <= 10]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not (c.lower().endswith("_id") or c.lower() == "id")]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    # If categorical column exists and >= 2 numeric columns exist, produce grouped bar chart across features
    if cat_cols and len(num_cols) >= 2:
        cat_c = category_col if category_col in cat_cols else cat_cols[0]
        feats = num_cols[:5]
        clean_labels = [c.replace("Cm", "").replace("_", " ").strip() for c in feats]
        grouped = df.groupby(cat_c)[feats].mean()
        fig, ax = plt.subplots(figsize=(9, 5.0))
        x = np.arange(len(clean_labels))
        n_cats = len(grouped)
        width = min(0.25, 0.75 / max(1, n_cats))
        for i, (cat_name, row) in enumerate(grouped.iterrows()):
            c = palette[i % len(palette)]
            ax.bar(x + (i - (n_cats - 1) / 2) * width, row.values, width, label=str(cat_name), color=c, edgecolor="white", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, fontsize=10)
        ax.set_ylabel("Average Measurement (cm)" if any("cm" in c.lower() for c in feats) else "Average Value", fontsize=10)
        ax.set_title(f"Average Measurements by {cat_c}", fontsize=12, fontweight="bold", pad=12)
        ax.legend(title=cat_c, frameon=True, fontsize=9.5)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"bar_{cat_c}_grouped.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return {"type": "bar_chart", "path": path, "columns": [cat_c] + feats}

    plot_df = df[[category_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    grouped = plot_df.groupby(category_col)[value_col].mean().dropna().sort_values(ascending=False)
    # If high cardinality, take top 12 to ensure legible presentation
    if len(grouped) > 12:
        grouped = grouped.head(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(grouped)), grouped.values, color=[palette[i % len(palette)] for i in range(len(grouped))], edgecolor="white", linewidth=0.7)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"Average {value_col} by {category_col}", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel(value_col, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"bar_{category_col}_{value_col}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"type": "bar_chart", "path": path, "columns": [category_col, value_col]}


def plot_line(df: pd.DataFrame, time_col: str, value_col: str, session_id: str) -> dict:
    out_dir = _ensure_output_dir(session_id)
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if 2 <= df[c].nunique(dropna=True) <= 10]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not (c.lower().endswith("_id") or c.lower() == "id")]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    # Check if time_col looks like datetime (avoid pandas converting floats to 1970 timestamps)
    is_date = False
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        is_date = True
    elif df[time_col].dtype == "object":
        sample = df[time_col].dropna().astype(str).str.strip()
        if len(sample) > 0 and pd.to_numeric(sample, errors="coerce").isna().sum() > len(sample) * 0.5:
            dt_s = pd.to_datetime(sample, errors="coerce")
            if dt_s.notna().sum() > len(sample) * 0.5:
                is_date = True
    if is_date:
        plot_df = df[[time_col, value_col]].dropna().copy()
        plot_df[time_col] = dt_series
        plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
        plot_df = plot_df.dropna().sort_values(time_col)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(plot_df[time_col], plot_df[value_col], color="#4C72B0", linewidth=2.0)
        ax.set_title(f"{value_col} over Time", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(time_col, fontsize=10)
        ax.set_ylabel(value_col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = os.path.join(out_dir, f"line_{time_col}_{value_col}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return {"type": "line_chart", "path": path, "columns": [time_col, value_col]}

    # Non-datetime profile by category (e.g. Average Measurement Profile by Species)
    if cat_cols and len(num_cols) >= 2:
        cat_c = cat_cols[0]
        feats = num_cols[:6]
        clean_labels = [c.replace("Cm", "").replace("_", " ").strip() for c in feats]
        grouped = df.groupby(cat_c)[feats].mean()
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for i, (cat_name, row) in enumerate(grouped.iterrows()):
            c = palette[i % len(palette)]
            ax.plot(clean_labels, row.values, marker="o", markersize=6, linewidth=2.2, label=str(cat_name), color=c)
        ax.set_title(f"Average Measurement Profile by {cat_c}", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylabel("Mean (cm)" if any("cm" in c.lower() for c in feats) else "Mean Value", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(title=cat_c, frameon=True, fontsize=9.5)
        fig.tight_layout()
        path = os.path.join(out_dir, f"line_{cat_c}_profile.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return {"type": "line_chart", "path": path, "columns": [cat_c] + feats}

    # Numeric sequence line
    plot_df = df[[time_col, value_col]].dropna().copy()
    plot_df[time_col] = pd.to_numeric(plot_df[time_col], errors="coerce")
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(time_col)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(plot_df[time_col], plot_df[value_col], color="#55A868", marker="o" if len(plot_df) < 50 else None, markersize=3)
    ax.set_title(f"{value_col} vs {time_col} Trend", fontsize=11, fontweight="bold")
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"line_{time_col}_{value_col}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"type": "line_chart", "path": path, "columns": [time_col, value_col]}


def plot_histogram(df: pd.DataFrame, col: str, session_id: str) -> dict:
    out_dir = _ensure_output_dir(session_id)
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if 2 <= df[c].nunique(dropna=True) <= 10]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not (c.lower().endswith("_id") or c.lower() == "id")]
    palette = {"setosa": "#4C72B0", "versicolor": "#DD8452", "virginica": "#55A868"}
    default_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    # Multi-panel distribution grid by category if category exists and >= 2 numeric columns
    if cat_cols and len(num_cols) >= 2:
        cat_c = cat_cols[0]
        cols_to_plot = num_cols[:4]
        n_plots = len(cols_to_plot)
        nrows = 2 if n_plots > 2 else 1
        ncols = 2 if n_plots >= 2 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 7.5) if n_plots > 2 else (9, 4.2))
        axes_flat = np.array(axes).flatten()
        categories = df[cat_c].dropna().unique()

        for idx, feat in enumerate(cols_to_plot):
            ax = axes_flat[idx]
            feat_clean = feat.replace("Cm", "").replace("_", " ").strip()
            feat_min = df[feat].min()
            feat_max = df[feat].max()
            bins = np.linspace(feat_min, feat_max, 18)

            for i, cat in enumerate(categories):
                sub = df[df[cat_c] == cat][feat].dropna()
                c = palette.get(str(cat).lower(), default_colors[i % len(default_colors)])
                ax.hist(sub, bins=bins, alpha=0.55, edgecolor="black", linewidth=0.8, label=str(cat), color=c)

            ax.set_title(feat_clean, fontsize=11, fontweight="medium")
            ax.set_xlabel("cm" if "cm" in feat.lower() else feat_clean, fontsize=9.5)
            ax.set_ylabel("Count", fontsize=9.5)
            ax.legend(title=cat_c, frameon=True, fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.suptitle(f"Distribution of Measurements by {cat_c}", fontsize=12.5, fontweight="bold", y=0.99)
        fig.tight_layout()
        path = os.path.join(out_dir, f"hist_{cat_c}_distributions.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return {"type": "histogram", "path": path, "columns": cols_to_plot}

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series.plot(kind="hist", bins=25, ax=ax, color="#4C72B0", edgecolor="black", linewidth=0.7)
    ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
    ax.set_xlabel(col)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"hist_{col}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"type": "histogram", "path": path, "columns": [col]}


def plot_scatter(df: pd.DataFrame, col_x: str, col_y: str, session_id: str) -> dict:
    out_dir = _ensure_output_dir(session_id)
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if 2 <= df[c].nunique(dropna=True) <= 10]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not (c.lower().endswith("_id") or c.lower() == "id")]
    palette = {"setosa": "#4C72B0", "versicolor": "#DD8452", "virginica": "#55A868"}
    default_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    # If categorical column exists and >= 4 numeric features exist, produce 2 side-by-side scatter plots
    if cat_cols and len(num_cols) >= 4:
        cat_c = cat_cols[0]
        categories = df[cat_c].dropna().unique()
        pair1_x = next((c for c in num_cols if "petallength" in c.lower() or "petal_length" in c.lower()), num_cols[2])
        pair1_y = next((c for c in num_cols if "petalwidth" in c.lower() or "petal_width" in c.lower()), num_cols[3])
        pair2_x = next((c for c in num_cols if "sepallength" in c.lower() or "sepal_length" in c.lower()), num_cols[0])
        pair2_y = next((c for c in num_cols if "sepalwidth" in c.lower() or "sepal_width" in c.lower()), num_cols[1])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
        for i, cat in enumerate(categories):
            sub = df[df[cat_c] == cat]
            c = palette.get(str(cat).lower(), default_colors[i % len(default_colors)])
            ax1.scatter(sub[pair1_x], sub[pair1_y], label=str(cat), color=c, alpha=0.85, s=38, edgecolors="white", linewidth=0.5)
            ax2.scatter(sub[pair2_x], sub[pair2_y], label=str(cat), color=c, alpha=0.85, s=38, edgecolors="white", linewidth=0.5)

        ax1.set_title(f"{pair1_x.replace('Cm','')} vs {pair1_y.replace('Cm','')}", fontsize=11, fontweight="medium")
        ax1.set_xlabel(pair1_x, fontsize=10)
        ax1.set_ylabel(pair1_y, fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(title=cat_c, frameon=True, fontsize=9)

        ax2.set_title(f"{pair2_x.replace('Cm','')} vs {pair2_y.replace('Cm','')}", fontsize=11, fontweight="medium")
        ax2.set_xlabel(pair2_x, fontsize=10)
        ax2.set_ylabel(pair2_y, fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(title=cat_c, frameon=True, fontsize=9)

        fig.tight_layout()
        path = os.path.join(out_dir, f"scatter_{pair1_x}_{pair2_x}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return {"type": "scatter_plot", "path": path, "columns": [pair1_x, pair1_y, pair2_x, pair2_y]}

    # Single scatter plot (with hue if category exists)
    plot_df = df[[col_x, col_y] + ([cat_cols[0]] if cat_cols else [])].copy()
    plot_df[col_x] = pd.to_numeric(plot_df[col_x], errors="coerce")
    plot_df[col_y] = pd.to_numeric(plot_df[col_y], errors="coerce")
    plot_df = plot_df.dropna()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if cat_cols:
        cat_c = cat_cols[0]
        categories = plot_df[cat_c].dropna().unique()
        for i, cat in enumerate(categories):
            sub = plot_df[plot_df[cat_c] == cat]
            c = palette.get(str(cat).lower(), default_colors[i % len(default_colors)])
            ax.scatter(sub[col_x], sub[col_y], label=str(cat), color=c, alpha=0.85, s=36, edgecolors="white", linewidth=0.5)
        ax.legend(title=cat_c, frameon=True, fontsize=9)
    else:
        ax.scatter(plot_df[col_x], plot_df[col_y], alpha=0.7, color="#4C72B0", s=32, edgecolors="white", linewidth=0.5)

    ax.set_title(f"{col_y} vs {col_x}", fontsize=11, fontweight="bold")
    ax.set_xlabel(col_x, fontsize=10)
    ax.set_ylabel(col_y, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"scatter_{col_x}_{col_y}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"type": "scatter_plot", "path": path, "columns": [col_x, col_y]}


def plot_heatmap(df: pd.DataFrame, session_id: str, max_cols: int = 12) -> dict:
    out_dir = _ensure_output_dir(session_id)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    for col in df.columns:
        if col not in numeric_df.columns:
            s_num = pd.to_numeric(df[col], errors="coerce")
            if s_num.notna().sum() >= len(df) * 0.5:
                numeric_df[col] = s_num

    id_like = [c for c in numeric_df.columns if c.lower().endswith("_id") or c.lower() == "id"]
    numeric_df = numeric_df.drop(columns=id_like)

    # Filter out columns with zero variance (constants or mostly empty)
    valid_cols = [c for c in numeric_df.columns if numeric_df[c].dropna().std() > 1e-5]
    numeric_df = numeric_df[valid_cols]

    if numeric_df.shape[1] < 2:
        return {}

    # If dataset has many columns, pick top most variable columns
    if numeric_df.shape[1] > max_cols:
        variances = numeric_df.var().sort_values(ascending=False)
        selected_cols = variances.head(max_cols).index.tolist()
        numeric_df = numeric_df[selected_cols]

    clean_labels = [c.replace("Cm", "").replace("_", " ").strip() for c in numeric_df.columns]
    corr = numeric_df.corr().round(2)
    n_vars = len(corr.columns)

    fig_size = max(6.2, min(9.5, n_vars * 0.8 + 1.2))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.92))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    font_size = 9.5 if n_vars <= 6 else (8.5 if n_vars <= 10 else 7.5)
    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    ax.set_xticklabels(clean_labels, rotation=0 if n_vars <= 4 else 35, ha="center" if n_vars <= 4 else "right", fontsize=font_size)
    ax.set_yticklabels(clean_labels, fontsize=font_size)

    for i in range(n_vars):
        for j in range(n_vars):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "#1e293b"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=font_size, fontweight="medium")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8.5)
    ax.set_title("Correlation Heatmap", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "heatmap.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"type": "heatmap", "path": path, "columns": list(corr.columns)}
    fig.savefig(path, dpi=130, bbox_inches="tight")
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
        if meta.get("is_numeric") and not (c.lower().endswith("_id") or c.lower() == "id")
    ]
    if not numeric_cols:
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if not (c.lower().endswith("_id") or c.lower() == "id")
        ]

    datetime_cols = [c for c, meta in column_profile.items() if meta.get("is_datetime")]
    if not datetime_cols:
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    categorical_cols = [
        c for c, meta in column_profile.items()
        if not meta.get("is_numeric") and not meta.get("is_datetime") and meta.get("n_unique", 999) <= 20
    ]
    if not categorical_cols:
        categorical_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if df[c].nunique(dropna=True) <= 20 and not (c.lower().endswith("_id") or c.lower() == "id")
        ]

    if datetime_cols and numeric_cols:
        recommendations.append({
            "chart_type": "line_chart",
            "columns": [datetime_cols[0], numeric_cols[0]],
            "rule_reason": "Dataset contains a time-based column paired with a numeric "
                            "column, indicating a trend-over-time relationship.",
        })
    elif len(numeric_cols) >= 2:
        recommendations.append({
            "chart_type": "line_chart",
            "columns": [numeric_cols[0], numeric_cols[1]],
            "rule_reason": f"Sequential trend relationship between '{numeric_cols[0]}' and '{numeric_cols[1]}'.",
        })

    if categorical_cols and numeric_cols:
        recommendations.append({
            "chart_type": "bar_chart",
            "columns": [categorical_cols[0], numeric_cols[0]],
            "rule_reason": "A categorical column and a numeric column suggest a "
                            "category comparison, best shown as a bar chart.",
        })
    elif numeric_cols:
        cat_candidates = [c for c in df.columns if df[c].nunique(dropna=True) <= 25 and not (c.lower().endswith("_id") or c.lower() == "id")]
        if cat_candidates:
            recommendations.append({
                "chart_type": "bar_chart",
                "columns": [cat_candidates[0], numeric_cols[0]],
                "rule_reason": "Comparison across discrete groups shown as a bar chart.",
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
