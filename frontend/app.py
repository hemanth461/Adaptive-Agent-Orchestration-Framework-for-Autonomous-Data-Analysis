"""
Streamlit Frontend: Adaptive Agent Orchestration Framework

Autonomous data analysis that dynamically selects between:
- Level 1 (Simple): Single General Agent
- Level 2 (Moderate): Tool-Augmented Agent
- Level 3 (Complex): Coordinated Multi-Agent Team (Data Prep -> Analysis -> Insights -> Validation)

Features:
- Professional, elegant, and modern dashboard UI
- Plain-English, easily understandable Data Quality report
- Completely removed EDA section as requested
- Plain-English, step-by-step Execution Trace explaining architecture choices
- Interactive Cleaned Dataset viewer and 1-click CSV download
- Preset demo datasets for frictionless 1-click evaluation
"""
import os
import json
import io
import time
import subprocess
import requests
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
SAMPLE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_datasets"))

# --- Auto-start Backend if running together in a unified deployment ---
@st.cache_resource
def _ensure_backend_running():
    if "127.0.0.1" in API_BASE or "localhost" in API_BASE:
        # Check if already running
        for _ in range(3):
            try:
                r = requests.get(f"{API_BASE}/health", timeout=1.0)
                if r.status_code == 200:
                    return True
            except Exception:
                time.sleep(1)

        # Fallback start if running standalone
        backend_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "backend"))
        if os.path.isdir(backend_dir):
            try:
                subprocess.Popen([
                    "python", "-m", "uvicorn", "app.api.main:app",
                    "--host", "127.0.0.1", "--port", "8000",
                    "--app-dir", backend_dir
                ])
                time.sleep(2.5)
            except Exception:
                pass
    return True

_ensure_backend_running()

# --- Page Config ---
st.set_page_config(
    page_title="Adaptive Agent Orchestration",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Modern Professional CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(165, 180, 252, 0.4);
        color: #c7d2fe;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.04);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Quality / Trace Action Cards */
    .action-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .action-card-success { border-left-color: #10b981; }
    .action-card-warning { border-left-color: #f59e0b; }
    .action-card-purple  { border-left-color: #8b5cf6; }

    .action-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .action-title {
        font-size: 0.98rem;
        font-weight: 600;
        color: #1e293b;
    }
    .action-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
    }
    .badge-blue { background: #eff6ff; color: #1d4ed8; }
    .badge-green { background: #ecfdf5; color: #047857; }
    .badge-amber { background: #fffbeb; color: #b45309; }
    .badge-purple { background: #f5f3ff; color: #6d28d9; }

    .action-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.55;
        margin: 0;
    }

    /* Insights Specific Styling */
    .takeaway-box {
        background: #f8fafc;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.5;
    }
    .evidence-pill {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        margin: 4px 6px 4px 0;
    }

    /* Architecture Callout Banner */
    .arch-banner {
        border-radius: 12px;
        padding: 20px 24px;
        margin: 18px 0 24px 0;
        border: 1px solid #e2e8f0;
    }
    .arch-simple {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #bbf7d0;
    }
    .arch-moderate {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-color: #bfdbfe;
    }
    .arch-complex {
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
        border-color: #ddd6fe;
    }
    
    /* Clean download button callout */
    .download-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 16px 0;
    }

    /* Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #475569;
        padding: 8px 16px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero Header ---
st.markdown(
    """
    <div class="hero-banner">
        <span class="hero-tag">AI Architecture Adaptor</span>
        <h1 class="hero-title">Adaptive Agent Orchestration Framework</h1>
        <p class="hero-subtitle">
            Autonomous data intelligence that assesses task complexity to dynamically route between 
            a <b>Single Fast Agent</b>, a <b>Tool-Augmented Agent</b>, or a <b>Coordinated Multi-Agent Team</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- State Management ---
if "result" not in st.session_state:
    st.session_state.result = None
if "active_query" not in st.session_state:
    st.session_state.active_query = ""
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []


# --- Preset Loaders ---
def load_sample_file(filename: str):
    path = os.path.join(SAMPLE_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


# --- Sidebar / Demo Control ---
with st.sidebar:
    st.markdown("### ⚡ Quick Demo Presets")
    st.caption("Click to load pre-configured scenarios and test adaptive routing:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🟢 Simple Run", use_container_width=True, help="Clean Sales Summary"):
            st.session_state.active_query = "Calculate average sales and create a bar chart by region."
            content = load_sample_file("sales_clean.csv")
            if content:
                st.session_state.loaded_files = [("sales_clean.csv", content)]
            st.rerun()

    with col_btn2:
        if st.button("🟡 Moderate Run", use_container_width=True, help="Marketing Imputation & Heatmap"):
            st.session_state.active_query = "Clean minor missing values in spend, analyze correlations between impressions, clicks, and spend, and generate a heatmap."
            content = load_sample_file("marketing_campaigns.csv")
            if content:
                st.session_state.loaded_files = [("marketing_campaigns.csv", content)]
            st.rerun()

    if st.button("🟣 Complex Multi-Agent Run", use_container_width=True, help="Customer Messy Data & Statistical Validation"):
        st.session_state.active_query = "Clean the dataset by fixing missing spend and duplicate customers. Compare annual spend across regions, visualize with charts, and validate whether regional differences are statistically significant."
        content = load_sample_file("customer_churn_spend.csv")
        if content:
            st.session_state.loaded_files = [("customer_churn_spend.csv", content)]
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Architecture Tiers")
    st.markdown(
        """
        - **Level 1 (Simple):** Score ≤ 3.5  
          *One General Agent* executes direct summarization.
        - **Level 2 (Moderate):** Score 3.6 – 6.5  
          *Tool-Augmented Agent* handles imputation & multi-charts.
        - **Level 3 (Complex):** Score ≥ 6.6  
          *Coordinated Multi-Agent Team* with fact-checking feedback loop.
        """
    )
    st.caption(f"Backend Server: `{API_BASE}`")


# --- Input Form ---
st.markdown("#### 📁 Upload Data & Define Task")
with st.container():
    uploaded_files = st.file_uploader(
        "Upload your dataset(s) (CSV format)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or multiple CSV files to analyze.",
    )

    # Use uploaded files if provided, otherwise check session state loaded files
    files_to_process = []
    if uploaded_files:
        files_to_process = [(f.name, f.getvalue()) for f in uploaded_files]
    elif st.session_state.loaded_files:
        files_to_process = st.session_state.loaded_files
        st.info(f"Using sample preset dataset: **{files_to_process[0][0]}** (click 'Clear' or upload your own to change)")

    query_input = st.text_area(
        "What questions or analysis do you want answered?",
        value=st.session_state.active_query,
        placeholder="e.g. Clean missing values, analyze spend across customer regions, create visualizations, and check if regional differences are statistically significant.",
        height=95,
    )

    col_sub, col_clr = st.columns([4, 1])
    with col_sub:
        run_submitted = st.button("🚀 Run Adaptive Analysis", type="primary", use_container_width=True)
    with col_clr:
        if st.button("Reset Form", use_container_width=True):
            st.session_state.result = None
            st.session_state.active_query = ""
            st.session_state.loaded_files = []
            st.rerun()

if run_submitted:
    if not files_to_process or not query_input.strip():
        st.error("Please provide at least one CSV file and enter an analysis request.")
    else:
        files_payload = [
            ("files", (fname, fcontent, "text/csv")) for fname, fcontent in files_to_process
        ]
        with st.spinner("Analyzing dataset & routing to the optimal architecture..."):
            try:
                response = requests.post(
                    f"{API_BASE}/analyze",
                    files=files_payload,
                    data={"query": query_input},
                    timeout=180,
                )
                response.raise_for_status()
                st.session_state.result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")

# ==============================================================================
# RESULTS DISPLAY
# ==============================================================================
result = st.session_state.result

if result:
    session_id = result.get("session_id", "")
    complexity_level = result.get("complexity_level", "simple").lower()
    complexity_score = result.get("complexity_score", 0.0)
    routing_reason = result.get("routing_reason", "")

    # Architecture style mappings
    arch_badges = {
        "simple": ("arch-simple", "Level 1: Single General Agent", "⚡ Ultra-Fast Single Agent Execution", "#15803d"),
        "moderate": ("arch-moderate", "Level 2: Tool-Augmented Agent", "🛠️ Tool-Augmented Single Agent", "#1d4ed8"),
        "complex": ("arch-complex", "Level 3: Multi-Agent Pipeline", "🤝 Coordinated Multi-Agent Team with Validation", "#6d28d9"),
    }
    css_class, tier_title, subtitle, badge_col = arch_badges.get(
        complexity_level, ("arch-simple", "Single Agent", "Standard Execution", "#15803d")
    )

    # --- Architecture Banner ---
    st.markdown(
        f"""
        <div class="arch-banner {css_class}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="background: {badge_col}; color: white; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
                        {tier_title}
                    </span>
                    <h3 style="margin: 10px 0 4px 0; color: #0f172a; font-weight: 700;">{subtitle}</h3>
                    <p style="margin: 0; color: #334155; font-size: 0.92rem; max-width: 760px;">
                        <b>Why this architecture:</b> {routing_reason}
                    </p>
                </div>
                <div style="background: white; border-radius: 10px; padding: 12px 18px; text-align: center; border: 1px solid rgba(0,0,0,0.06); min-width: 140px;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Complexity Score</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {badge_col};">{complexity_score}<span style="font-size: 1rem; color: #94a3b8;">/10</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Quick Top Metric Tiles ---
    profile = result.get("dataset_profile", {})
    ds_list = profile.get("datasets", [])
    total_rows = sum(d.get("n_rows", 0) for d in ds_list)
    total_cols = max((d.get("n_columns", 0) for d in ds_list), default=0)
    cleaned_rows = result.get("cleaned_rows", total_rows)
    n_cleaning_actions = len(result.get("cleaning_log", []))
    n_charts = len(result.get("visualizations", []))
    n_insights = len(result.get("validation_results", []))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Input Records</div>
                <div class="metric-value">{total_rows:,}</div>
                <div class="metric-sub">{len(ds_list)} dataset(s) analyzed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Data Health Fixes</div>
                <div class="metric-value">{n_cleaning_actions}</div>
                <div class="metric-sub">Auto-remediated issues</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Cleaned Rows</div>
                <div class="metric-value">{cleaned_rows:,}</div>
                <div class="metric-sub">Ready for export</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Charts Created</div>
                <div class="metric-value">{n_charts}</div>
                <div class="metric-sub">Semantic type selection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Validated Claims</div>
                <div class="metric-value">{n_insights}</div>
                <div class="metric-sub">Fact-checked findings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --- Navigation Tabs (EDA REMOVED, CLEANED DATASET ADDED) ---
    tabs = st.tabs([
        "📋 Overview",
        "🧹 Data Quality",
        "📥 Cleaned Dataset",
        "📈 Visualizations",
        "💡 Key Insights & Validation",
        "⏱️ Execution Trace",
        "📑 Final Report",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 📊 Dataset Overview")
        st.caption("Initial health and structural metrics observed before any automated transformations.")

        for ds in ds_list:
            st.markdown(f"#### 📄 File: `{ds['name']}`")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{ds['n_rows']:,}")
            col2.metric("Total Columns", ds["n_columns"])
            col3.metric("Missing Data", f"{ds['missing_pct']}%")
            col4.metric("Duplicate Rows", ds["duplicate_count"])

            with st.expander(f"View Columns & Data Types ({ds['name']})", expanded=False):
                col_data = []
                for cname, cinfo in ds["columns"].items():
                    col_data.append({
                        "Column Name": cname,
                        "Data Type": cinfo.get("dtype"),
                        "Missing (%)": f"{cinfo.get('missing_pct', 0)}%",
                        "Sample Values": ", ".join(str(v) for v in cinfo.get("sample_values", [])[:3]),
                    })
                st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: DATA QUALITY (PLAIN-ENGLISH, EASILY UNDERSTANDABLE)
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 🧹 Data Quality & Automatic Cleaning")
        st.caption("Clear, plain-English summary of all issues detected and how the system safely resolved them.")

        cleaning_log = result.get("cleaning_log", [])

        # Categorize actions for non-technical viewers
        missing_fixes = [e for e in cleaning_log if e.get("issue_type") == "missing"]
        dup_fixes = [e for e in cleaning_log if e.get("issue_type") == "duplicate"]
        dtype_fixes = [e for e in cleaning_log if e.get("issue_type") == "dtype"]
        outlier_flags = [e for e in cleaning_log if e.get("issue_type") == "outlier"]

        # Health Summary Pills
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">🩹 Missing Values Fixed</div>
                    <div class="metric-value">{len(missing_fixes)}</div>
                    <div class="metric-sub">Columns imputed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">🧹 Duplicates Removed</div>
                    <div class="metric-value">{len(dup_fixes)}</div>
                    <div class="metric-sub">Redundant row sets</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">🔄 Types Standardized</div>
                    <div class="metric-value">{len(dtype_fixes)}</div>
                    <div class="metric-sub">Text to number conversions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">⚠️ Outliers Monitored</div>
                    <div class="metric-value">{len(outlier_flags)}</div>
                    <div class="metric-sub">Flagged (none erased)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        if cleaning_log:
            st.markdown("#### 🔍 Plain-English Actions Taken")

            for entry in cleaning_log:
                itype = entry.get("issue_type", "issue")
                col_name = entry.get("column_name")
                raw_reason = entry.get("reason", "")
                strategy = entry.get("strategy_applied", "")

                # Translate into crystal clear, non-technical points
                if itype == "missing":
                    title = f"Filled Missing Information in '{col_name}'"
                    badge_text = "🩹 Missing Values Restored"
                    badge_class = "badge-blue"
                    card_class = "action-card"
                    desc = (
                        f"We discovered blank or missing values in <b>{col_name}</b>. "
                        f"To keep calculations accurate without discarding valuable records, "
                        f"we automatically filled in the blanks with typical representative values. "
                        f"<i>(Applied: {strategy.replace('_', ' ')}).</i>"
                    )

                elif itype == "duplicate":
                    title = "Removed Exact Duplicate Rows"
                    badge_text = "🧹 Duplicates Cleaned"
                    badge_class = "badge-green"
                    card_class = "action-card action-card-success"
                    desc = (
                        f"Found and safely removed repeated identical records. "
                        f"This ensures no customer, transaction, or measurement is accidentally double-counted in your results."
                    )

                elif itype == "dtype":
                    title = f"Converted Text Format to Numbers in '{col_name}'"
                    badge_text = "🔄 Format Standardized"
                    badge_class = "badge-purple"
                    card_class = "action-card action-card-purple"
                    desc = (
                        f"Numbers in column <b>{col_name}</b> were stored as text. "
                        f"We converted them into genuine numerical values so formulas, charts, and trends calculate properly."
                    )

                elif itype == "outlier":
                    title = f"Flagged Extreme / Unusual Values in '{col_name}'"
                    badge_text = "⚠️ Outlier Detected"
                    badge_class = "badge-amber"
                    card_class = "action-card action-card-warning"
                    desc = (
                        f"Detected unusually high or low numbers in <b>{col_name}</b>. "
                        f"<b>Your original data was preserved intact</b> (nothing was deleted), "
                        f"but these extreme points are flagged so you can review them."
                    )
                else:
                    title = f"Data Adjustment on '{col_name or 'dataset'}'"
                    badge_text = "⚙️ Processed"
                    badge_class = "badge-blue"
                    card_class = "action-card"
                    desc = raw_reason

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="action-header">
                            <span class="action-title">{title}</span>
                            <span class="action-badge {badge_class}">{badge_text}</span>
                        </div>
                        <p class="action-desc">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Quick download shortcut
            st.markdown(
                """
                <div class="download-box">
                    <h4 style="margin: 0 0 6px 0; color: #0f172a;">Ready to export your clean data?</h4>
                    <p style="color: #64748b; font-size: 0.88rem; margin: 0 0 12px 0;">
                        All corrections have been saved. Head to the <b>Cleaned Dataset</b> tab to inspect and download your CSV.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("🔧 Technical Cleaning Audit (For Engineers & Evaluators)", expanded=False):
                st.caption("Raw strategy names and parameters logged by Data Preparation Agent:")
                st.table(cleaning_log)

        else:
            st.success("✨ **Pristine Data Quality:** Your dataset was already clean! No missing cells, duplicate rows, or formatting issues were detected.")

    # --------------------------------------------------------------------------
    # TAB 3: CLEANED DATASET (PRODUCE AND DOWNLOAD CLEANED CSV)
    # --------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 📥 Cleaned Dataset Deliverable")
        st.caption("Inspect the fully remediated dataset and download the clean CSV file for use in Excel, BI dashboards, or machine learning.")

        csv_available = result.get("cleaned_csv_available", False)
        csv_text = result.get("cleaned_csv_text", "")
        cleaned_preview = result.get("cleaned_preview", [])
        cleaned_filename = result.get("cleaned_csv_filename", f"cleaned_dataset_{session_id[:8]}.csv")

        if csv_available or csv_text or cleaned_preview:
            # Action banner with prominent download
            col_down1, col_down2 = st.columns([3, 2])
            with col_down1:
                st.markdown(
                    f"""
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">
                        <div style="color: #166534; font-weight: 700; font-size: 1rem;">✅ Cleaned Dataset Ready</div>
                        <div style="color: #15803d; font-size: 0.85rem; margin-top: 2px;">
                            Total rows: <b>{cleaned_rows:,}</b> | Total columns: <b>{len(result.get('cleaned_columns', []))}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_down2:
                if csv_text:
                    st.download_button(
                        label="⬇️ Download Cleaned CSV File",
                        data=csv_text,
                        file_name=f"cleaned_dataset_{session_id[:8]}.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )
                else:
                    try:
                        csv_resp = requests.get(f"{API_BASE}/dataset/{session_id}/download-cleaned", timeout=10)
                        if csv_resp.status_code == 200:
                            st.download_button(
                                label="⬇️ Download Cleaned CSV File",
                                data=csv_resp.content,
                                file_name=f"cleaned_dataset_{session_id[:8]}.csv",
                                mime="text/csv",
                                type="primary",
                                use_container_width=True,
                            )
                        else:
                            st.caption("Cleaned CSV currently unavailable.")
                    except Exception:
                        st.caption("Cleaned CSV currently unavailable.")

            # Table preview
            st.markdown("#### 📋 Cleaned Data Preview")
            if cleaned_preview:
                preview_df = pd.DataFrame(cleaned_preview)
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                st.caption(f"Showing first {len(preview_df)} records of {cleaned_rows:,} total clean rows.")
            elif csv_text:
                preview_df = pd.read_csv(io.StringIO(csv_text))
                st.dataframe(preview_df.head(25), use_container_width=True, hide_index=True)
                st.caption(f"Showing first 25 records of {cleaned_rows:,} total clean rows.")
            else:
                st.info("Dataset was processed. Use the download button above to retrieve the complete CSV.")
        else:
            st.info("No cleaned dataset available for this session.")

    # --------------------------------------------------------------------------
    # TAB 4: VISUALIZATIONS
    # --------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### 📈 Visualizations")
        st.caption("Charts generated based on variable data types, cardinality, and task requirements.")

        vizs = result.get("visualizations", [])
        if vizs:
            for i, v in enumerate(vizs):
                filename = os.path.basename(v.get("path", ""))
                chart_name = v.get("type", "chart").replace("_", " ").title()
                cols = ", ".join(v.get("columns", []))

                # Load image from local disk path or fetch bytes server-side
                img_data = None
                img_path = v.get("path")
                if img_path and os.path.isfile(img_path):
                    img_data = img_path
                else:
                    try:
                        resp = requests.get(f"{API_BASE}/chart/{session_id}/{filename}", timeout=6)
                        if resp.status_code == 200:
                            img_data = resp.content
                    except Exception:
                        pass

                with st.container():
                    st.markdown(f"#### 📊 {chart_name} ({cols})")
                    if img_data:
                        st.image(img_data, use_container_width=True)
                    else:
                        st.warning(f"Could not load chart image for {filename}.")
                    st.caption(f"**Design Rationale:** {v.get('reason', '')}")
                    st.markdown("---")
        else:
            st.write("No visualizations were requested or generated for this task.")

    # --------------------------------------------------------------------------
    # TAB 5: KEY INSIGHTS & VALIDATION
    # --------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("### 💡 Key Findings & Statistical Fact-Checking")
        st.caption("Deep analytical findings generated by AI agents and verified via statistical testing before delivery.")

        validations = result.get("validation_results", [])
        if validations:
            n_total = len(validations)
            n_supported = sum(1 for v in validations if v.get("supported"))
            n_trends = n_total - n_supported

            # Mini KPI Bar for Insights
            kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
            with kpi_c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Findings</div>
                        <div class="metric-value">{n_total}</div>
                        <div class="metric-sub">Synthesized observations</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi_c2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Statistically Supported</div>
                        <div class="metric-value" style="color: #10b981;">{n_supported}</div>
                        <div class="metric-sub">Passed hypothesis checks</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi_c3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Observed Trends</div>
                        <div class="metric-value" style="color: #f59e0b;">{n_trends}</div>
                        <div class="metric-sub">Directional / caveat findings</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            for i, v in enumerate(validations):
                is_supported = v.get("supported", False)
                title = v.get("title") or f"Finding {i+1}"
                claim = v.get("claim", "")
                claim_type = v.get("claim_type", "general")
                evidence = v.get("evidence", {})
                takeaway = v.get("business_takeaway", "")
                test_used = v.get("test_used", "Empirical check")
                sample_n = v.get("n", "N/A")
                reason = v.get("reason", "")

                # Category badge styling
                type_labels = {
                    "group_comparison": ("📊 Segment Comparison", "badge-blue"),
                    "correlation": ("🔗 Correlation Driver", "badge-purple"),
                    "general": ("📈 Portfolio Benchmark", "badge-blue"),
                    "distribution": ("📊 Distribution Health", "badge-blue"),
                }
                type_name, type_badge_class = type_labels.get(claim_type, ("💡 Analytical Finding", "badge-blue"))

                if is_supported:
                    badge_html = '<span class="action-badge badge-green">✅ Statistically Supported</span>'
                    card_class = "action-card action-card-success"
                else:
                    badge_html = '<span class="action-badge badge-amber">⚠️ Observed Trend / Caveat</span>'
                    card_class = "action-card action-card-warning"

                # Build Evidence Pills
                evidence_html = ""
                if isinstance(evidence, dict) and evidence:
                    pills = []
                    for k, val in evidence.items():
                        if k.endswith("_mean"):
                            grp = k.replace("_mean", "")
                            pills.append(f'<span class="evidence-pill">{grp}: <b>${float(val):,.2f}</b></span>')
                        elif k == "correlation":
                            pills.append(f'<span class="evidence-pill">Pearson r: <b>{float(val):.3f}</b></span>')
                        elif k in ["mean", "median", "std"]:
                            pills.append(f'<span class="evidence-pill">{k.capitalize()}: <b>{float(val):,.2f}</b></span>')
                    if pills:
                        evidence_html = f'<div style="margin: 8px 0 4px 0;">{" ".join(pills)}</div>'

                # Build Strategic Takeaway Box
                takeaway_html = ""
                if takeaway:
                    takeaway_html = f"""
                    <div class="takeaway-box">
                        <b>💡 Strategic Recommendation:</b> {takeaway}
                    </div>
                    """

                st.markdown(
                    f"""
                    <div class="{card_class}" style="margin-bottom: 18px; padding: 18px 22px;">
                        <div class="action-header" style="margin-bottom: 6px;">
                            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                <span class="action-badge {type_badge_class}">{type_name}</span>
                                <span style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">{title}</span>
                            </div>
                            {badge_html}
                        </div>
                        <div style="font-size: 0.94rem; color: #1e293b; font-weight: 500; line-height: 1.6; margin: 8px 0;">
                            {claim}
                        </div>
                        {evidence_html}
                        <div style="background: #f8fafc; border-radius: 6px; padding: 8px 12px; margin-top: 8px; font-size: 0.8rem; color: #475569;">
                            <b>Validation Test:</b> <code>{test_used}</code> &nbsp;|&nbsp; 
                            <b>Sample Size:</b> <code>n={sample_n}</code> &nbsp;|&nbsp; 
                            <b>Reasoning:</b> {reason}
                        </div>
                        {takeaway_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No formal hypothesis validation was triggered for this complexity level.")

    # --------------------------------------------------------------------------
    # TAB 6: EXECUTION TRACE (EASILY UNDERSTANDABLE POINTS)
    # --------------------------------------------------------------------------
    with tabs[5]:
        st.markdown("### ⏱️ Execution Trace & Architecture Explainability")
        st.caption("A human-readable timeline showing every decision the system made and why — the core differentiator of adaptive orchestration.")

        trace = result.get("execution_trace", [])

        # Explainability Core Summary Box
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #475569; text-transform: uppercase;">
                    Decision Trace Overview
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #0f172a; margin: 4px 0;">
                    Task classified as <b>{complexity_level.upper()}</b> (Score: {complexity_score}/10)
                </div>
                <p style="margin: 0; font-size: 0.9rem; color: #64748b;">
                    Instead of forcing a heavy multi-agent pipeline on every dataset, the orchestrator evaluated the operations 
                    needed and picked the leanest architecture that could solve the request reliably.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action translations to plain English
        action_translations = {
            "dataset_profiling": ("🔍 Step 1: Scanned Dataset Integrity", "Scanned the dataset to count rows, columns, and measure missing values and duplicates.", "badge-blue"),
            "task_understanding": ("🧠 Step 2: Interpreted User Goal", "Analyzed your prompt to identify all requested analytical operations.", "badge-purple"),
            "complexity_assessment": ("⚖️ Step 3: Evaluated Task Complexity", "Scored dataset difficulty and analytical demands to determine the leanest agent architecture.", "badge-purple"),
            "routing_decision": ("🧭 Step 4: Routed to Optimal Architecture", "Chose the best execution path to avoid unnecessary agent delays and token waste.", "badge-green"),
            "handle_missing_values": ("🩹 Data Cleaning: Missing Values", "Imputed missing cells with representative distribution values.", "badge-blue"),
            "handle_duplicates": ("🧹 Data Cleaning: Deduplication", "Detected and safely removed repeated identical records.", "badge-blue"),
            "fix_dtype_issues": ("🔄 Data Cleaning: Format Standardization", "Converted text-formatted numbers into standard numeric data types.", "badge-blue"),
            "detect_outliers": ("⚠️ Data Quality: Outlier Check", "Identified extreme values and flagged them without deleting real data.", "badge-amber"),
            "descriptive_stats": ("📊 Statistical Computation", "Calculated averages, medians, standard deviations, and distributions.", "badge-blue"),
            "correlation_analysis": ("🔗 Correlation Analysis", "Computed relationships and mutual dependencies between metrics.", "badge-blue"),
            "chart_selection": ("📈 Automated Chart Selection", "Selected optimal visualization types matching data semantics.", "badge-green"),
            "generate_insights": ("💡 Finding Discovery", "Formulated observations and findings backed by calculated numbers.", "badge-purple"),
            "validate_insights": ("🛡️ Statistical Fact-Checking", "Subjected findings to statistical tests (t-test / ANOVA / correlation significance).", "badge-green"),
            "request_retry": ("🔄 Feedback Loop: Retry Requested", "Weak claim detected; looped back once to verify numbers.", "badge-amber"),
            "generate_final_report": ("📑 Deliverable Compilation", "Assembled the executive summary, findings, and visualizations into the final report.", "badge-blue"),
        }

        if trace:
            for step in trace:
                act = step.get("action", "")
                agent = step.get("agent_name", "")
                raw_reason = step.get("reason", "")
                duration_ms = step.get("duration_ms", 0)
                is_llm = step.get("llm_call", False)

                friendly_title, friendly_desc, badge_style = action_translations.get(
                    act, (f"Action: {act.replace('_', ' ').title()}", raw_reason, "badge-blue")
                )

                method_badge = '<span class="action-badge badge-purple">🤖 AI Reasoning</span>' if is_llm else '<span class="action-badge badge-blue">⚡ Deterministic Rule</span>'

                st.markdown(
                    f"""
                    <div class="action-card">
                        <div class="action-header">
                            <span class="action-title">[{step['step_order']}] {friendly_title}</span>
                            <div style="display: flex; gap: 6px; align-items: center;">
                                {method_badge}
                                <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">{duration_ms} ms</span>
                            </div>
                        </div>
                        <p class="action-desc">{friendly_desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"Developer Audit Log (Step {step['step_order']})", expanded=False):
                    st.code(
                        f"Agent: {agent}\nAction: {act}\nTool: {step.get('tool_used')}\nReason: {raw_reason}",
                        language="yaml",
                    )
        else:
            st.info("No trace steps recorded.")

    # --------------------------------------------------------------------------
    # TAB 7: FINAL REPORT
    # --------------------------------------------------------------------------
    with tabs[6]:
        st.markdown("### 📑 Final Executive Report")
        st.caption("Complete synthesized analytical findings and conclusions.")

        report_md = result.get("final_report", "")
        if report_md:
            st.markdown(report_md)

            st.write("")
            col_rep1, col_rep2 = st.columns(2)
            with col_rep1:
                st.download_button(
                    "📄 Download Report (.md)",
                    report_md,
                    file_name=f"analysis_report_{session_id[:8]}.md",
                    use_container_width=True,
                )
            with col_rep2:
                try:
                    pdf_response = requests.get(f"{API_BASE}/report/{session_id}/download", timeout=15)
                    if pdf_response.status_code == 200:
                        st.download_button(
                            "📑 Download Full Report (PDF with Embedded Charts)",
                            pdf_response.content,
                            file_name=f"analysis_report_{session_id[:8]}.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                        )
                    else:
                        st.caption("PDF generation is finalizing.")
                except Exception:
                    st.caption("Backend PDF service currently unreachable.")
        else:
            st.info("Report not yet generated.")
