"""
PDF Report Generator.

Extends Module M10 (Report Generation) with a genuine downloadable
deliverable: a PDF containing all report sections AND the actual chart
images embedded inline -- not just their names/reasons, which is all the
markdown report shows in the Streamlit UI.

Uses fpdf2: pure-Python, no system dependencies (no wkhtmltopdf/weasyprint
needed), so it installs cleanly on Windows without extra tooling.
"""
import os
from fpdf import FPDF

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "outputs")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "Data Analysis Report", ln=True)
        self.set_draw_color(200, 200, 200)
        self.line(10, 20, 200, 20)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(20, 20, 20)
        self.set_x(self.l_margin)
        self.ln(3)
        safe_text = str(text).encode("latin-1", errors="replace").decode("latin-1")
        self.cell(0, 8, safe_text, ln=True)
        self.set_x(self.l_margin)
        self.set_text_color(60, 60, 60)
        self.set_font("Helvetica", "", 10)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.set_x(self.l_margin)
        # fpdf2's multi_cell needs latin-1-safe text; replace unsupported chars
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        self.multi_cell(0, 6, safe_text)
        self.set_x(self.l_margin)
        self.ln(1)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.set_x(self.l_margin)
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        self.multi_cell(0, 6, f"-  {safe_text}")
        self.set_x(self.l_margin)


def generate_pdf_report(state: dict) -> str:
    """
    Builds a complete PDF report from the final AgentState and saves it to
    data/outputs/{session_id}/report.pdf.

    Includes: executive summary, architecture decision + reasoning, dataset
    overview, data quality/cleaning log, analysis results, EVERY generated
    chart embedded as an actual image (not just named), insights with
    validation status, and the execution trace summary.

    Returns the saved file path.
    """
    session_id = state["session_id"]
    out_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # --- Request + Executive Summary ---
    pdf.section_title("Request")
    pdf.body_text(state.get("user_query", ""))

    pdf.section_title("Executive Summary")
    exec_summary = _extract_exec_summary(state.get("final_report", ""))
    pdf.body_text(exec_summary or "No summary available.")

    # --- Architecture Decision ---
    pdf.section_title("Architecture Decision")
    pdf.bullet(f"Complexity Score: {state.get('complexity_score', 'N/A')}/10")
    pdf.bullet(f"Level Selected: {str(state.get('complexity_level', 'N/A')).capitalize()}")
    pdf.bullet(f"LLM Tiebreaker Used: {state.get('used_llm_tiebreaker', False)}")
    pdf.body_text(state.get("routing_reason", ""))

    # --- Dataset Overview ---
    profile = state.get("dataset_profile", {})
    if profile.get("datasets"):
        pdf.section_title("Dataset Overview")
        for ds in profile["datasets"]:
            pdf.bullet(
                f"{ds['name']}: {ds['n_rows']} rows x {ds['n_columns']} columns, "
                f"{ds['missing_pct']}% missing, {ds['duplicate_count']} duplicate rows"
            )

    # --- Data Quality & Preprocessing ---
    cleaning_log = state.get("cleaning_log", [])
    if cleaning_log:
        pdf.section_title("Data Quality & Preprocessing")
        for entry in cleaning_log:
            col = entry.get("column_name") or "(table-level)"
            pdf.bullet(f"{col} [{entry['issue_type']}]: {entry['reason']}")

    # --- Analysis Results ---
    analysis_results = state.get("analysis_results", {})
    if analysis_results.get("descriptive_stats"):
        pdf.section_title("Descriptive Statistics")
        for col, stats in analysis_results["descriptive_stats"].items():
            pdf.bullet(
                f"{col}: mean={stats['mean']}, median={stats['median']}, std={stats['std']}"
            )

    if analysis_results.get("group_comparison"):
        gc = analysis_results["group_comparison"]
        pdf.section_title(f"Group Comparison: {gc['value_col']} by {gc['group_col']}")
        for group, vals in gc["result"].items():
            pdf.bullet(f"{group}: mean={vals.get('mean')}, n={vals.get('count')}")

    # --- Visualizations: THE actual embedded images, not just names ---
    visualizations = state.get("visualizations", [])
    if visualizations:
        for v in visualizations:
            img_path = v.get("path")
            if not img_path or not os.path.isfile(img_path):
                continue
            pdf.add_page()
            pdf.section_title(f"Chart: {v['type'].replace('_', ' ').title()}")
            pdf.body_text(v.get("reason", ""))
            pdf.ln(2)
            # Fit image to page width with margins
            page_width = pdf.w - 20
            pdf.image(img_path, x=10, w=page_width)

    # --- Insights & Validation ---
    validation_results = state.get("validation_results", [])
    if validation_results:
        pdf.add_page()
        pdf.section_title("Insights & Validation")
        for v in validation_results:
            status = "SUPPORTED" if v["supported"] else "UNSUPPORTED (caveat)"
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(pdf.l_margin)
            safe_claim = v["claim"].encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, f"{safe_claim}  [{status}]")
            pdf.set_x(pdf.l_margin)
            pdf.body_text(v.get("reason", ""))

    # --- Execution Summary ---
    trace = state.get("execution_trace", [])
    if trace:
        pdf.section_title("Execution Summary")
        n_llm_calls = sum(1 for t in trace if t.get("llm_call"))
        pdf.bullet(f"Total steps: {len(trace)}")
        pdf.bullet(f"LLM calls made: {n_llm_calls}")
        pdf.bullet(f"Retry count: {state.get('retry_count', 0)}")

    pdf.output(pdf_path)
    return pdf_path


def _extract_exec_summary(markdown_report: str) -> str:
    """Pull just the executive summary paragraph out of the markdown report."""
    if "## Executive Summary" not in markdown_report:
        return ""
    after = markdown_report.split("## Executive Summary", 1)[1]
    # summary ends at the next "## " heading
    return after.split("## ", 1)[0].strip()
