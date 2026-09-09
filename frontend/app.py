"""
Streamlit Frontend.

Module: M11 (Streamlit Frontend)

Talks to the FastAPI backend via HTTP. Run the backend first:
    cd backend && uvicorn app.api.main:app --reload

Then run this:
    streamlit run frontend/app.py
"""
import os
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Adaptive Agent Orchestration", layout="wide")
st.title("🧭 Adaptive Agent Orchestration Framework")
st.caption("Autonomous data analysis that adapts its own architecture to task complexity.")

# --- Session state ---
if "result" not in st.session_state:
    st.session_state.result = None

# --- Upload form ---
with st.form("analyze_form"):
    uploaded_files = st.file_uploader(
        "Upload dataset(s) (CSV)", type=["csv"], accept_multiple_files=True
    )
    query = st.text_area(
        "What would you like to analyze?",
        placeholder="e.g. Clean the data, compare spend by region, and validate whether the difference is significant.",
    )
    submitted = st.form_submit_button("Run Analysis")

if submitted:
    if not uploaded_files or not query.strip():
        st.error("Please upload at least one CSV file and enter a query.")
    else:
        files_payload = [
            ("files", (f.name, f.getvalue(), "text/csv")) for f in uploaded_files
        ]
        with st.spinner("Running adaptive analysis pipeline..."):
            try:
                response = requests.post(
                    f"{API_BASE}/analyze",
                    files=files_payload,
                    data={"query": query},
                    timeout=180,
                )
                response.raise_for_status()
                st.session_state.result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

# --- Results ---
result = st.session_state.result
if result:
    session_id = result.get("session_id", "")

    st.success(
        f"Complexity: **{result.get('complexity_level', 'N/A').upper()}** "
        f"(score {result.get('complexity_score', 'N/A')}/10)"
    )
    st.info(result.get("routing_reason", ""))

    tabs = st.tabs([
        "Overview", "Data Quality", "EDA", "Visualizations",
        "Insights", "Execution Trace", "Report",
    ])

    # --- Overview ---
    with tabs[0]:
        st.subheader("Dataset Overview")
        profile = result.get("dataset_profile", {})
        for ds in profile.get("datasets", []):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", ds["n_rows"])
            col2.metric("Columns", ds["n_columns"])
            col3.metric("Missing %", f"{ds['missing_pct']}%")
            col4.metric("Duplicates", ds["duplicate_count"])
            with st.expander(f"Column details: {ds['name']}"):
                st.json(ds["columns"])

    # --- Data Quality ---
    with tabs[1]:
        st.subheader("Data Quality & Preprocessing")
        cleaning_log = result.get("cleaning_log", [])
        if cleaning_log:
            for entry in cleaning_log:
                col = entry.get("column_name") or "(table-level)"
                st.markdown(f"**{col}** — `{entry['issue_type']}` → `{entry['strategy_applied']}`")
                st.caption(entry["reason"])
        else:
            st.write("No cleaning was required or performed for this task level.")

    # --- EDA ---
    with tabs[2]:
        st.subheader("Analysis Results")
        analysis = result.get("analysis_results", {})
        if "descriptive_stats" in analysis:
            st.markdown("**Descriptive Statistics**")
            st.json(analysis["descriptive_stats"])
        if "correlation_matrix" in analysis:
            st.markdown("**Correlation Matrix**")
            st.json(analysis["correlation_matrix"])
        if "group_comparison" in analysis:
            st.markdown("**Group Comparison**")
            st.json(analysis["group_comparison"])
        if not analysis:
            st.write("No analysis results for this task level.")

    # --- Visualizations ---
    with tabs[3]:
        st.subheader("Visualizations")
        vizs = result.get("visualizations", [])
        if vizs:
            for v in vizs:
                filename = os.path.basename(v["path"])
                img_url = f"{API_BASE}/chart/{session_id}/{filename}"
                st.image(img_url, caption=v["reason"])
        else:
            st.write("No visualizations were generated for this task.")

    # --- Insights ---
    with tabs[4]:
        st.subheader("Insights & Validation")
        validations = result.get("validation_results", [])
        if validations:
            for v in validations:
                status = "✅ Supported" if v["supported"] else "⚠️ Unsupported (caveat)"
                st.markdown(f"**{v['claim']}** — {status}")
                st.caption(f"Test: {v.get('test_used', 'N/A')} | {v['reason']}")
        else:
            st.write("No insights were generated for this task level.")

    # --- Execution Trace ---
    with tabs[5]:
        st.subheader("Execution Trace")
        st.caption("Every agent decision, in order, with its reason — this is the system's explainability layer.")
        trace = result.get("execution_trace", [])
        for step in trace:
            llm_badge = "🧠 LLM" if step.get("llm_call") else "⚙️ Rule"
            st.markdown(f"**[{step['step_order']}] {step['agent_name']}** — `{step['action']}` {llm_badge}")
            st.caption(step["reason"])

    # --- Report ---
    with tabs[6]:
        st.subheader("Final Report")
        report_md = result.get("final_report", "")
        if report_md:
            st.markdown(report_md)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Report (.md)", report_md, file_name="analysis_report.md"
                )
            with col2:
                try:
                    pdf_response = requests.get(
                        f"{API_BASE}/report/{session_id}/download", timeout=30
                    )
                    if pdf_response.status_code == 200:
                        st.download_button(
                            "Download Full Report (PDF, with charts)",
                            pdf_response.content,
                            file_name=f"analysis_report_{session_id[:8]}.pdf",
                            mime="application/pdf",
                        )
                    else:
                        st.caption("PDF report not available for this session.")
                except requests.exceptions.RequestException:
                    st.caption("Could not reach backend to fetch the PDF report.")
        else:
            st.write("Report not yet generated.")
