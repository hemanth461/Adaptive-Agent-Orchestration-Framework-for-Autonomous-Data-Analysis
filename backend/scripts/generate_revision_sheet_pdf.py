import os
import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class RevisionSheetPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(14, 16, 14)
        self.set_auto_page_break(auto=True, margin=16)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 116, 139)
            self.cell(0, 6, "Adaptive Agent Orchestration Framework -- Interview Revision Sheet (v2)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_draw_color(226, 232, 240)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Prepared for Interview Preparation  |  Page {self.page_no()} of {{nb}}", align="C")

    def section_banner(self, title):
        # Ensure we don't orphan section banner at the very bottom
        if self.get_y() > self.h - 45:
            self.add_page()
        self.ln(3)
        self.set_fill_color(30, 58, 138)  # Deep Navy Blue
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(self.epw, 7.5, f"  {title.upper()}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def subsection_title(self, text):
        if self.get_y() > self.h - 35:
            self.add_page()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 58, 138)
        self.cell(self.epw, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def qa_pair(self, question, answer):
        # Calculate approximate height needed to avoid awkward orphan questions
        q_lines = len(question) // 80 + 1
        a_lines = len(answer) // 90 + answer.count('\n') + 1
        est_height = q_lines * 4.5 + a_lines * 4.0 + 8
        if self.get_y() + est_height > self.h - 18 and est_height < 70:
            self.add_page()

        # Question
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(15, 23, 42)
        clean_q = self._clean_text(question)
        self.multi_cell(self.epw, 4.6, clean_q, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

        # Answer
        self.set_font("Helvetica", "", 8.8)
        self.set_text_color(51, 65, 85)
        clean_a = self._clean_text(answer)
        self.multi_cell(self.epw, 4.2, clean_a, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Divider line between Q&A pairs
        self.ln(2.5)
        self.set_draw_color(241, 245, 249)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2.5)

    def body_paragraph(self, text):
        self.set_font("Helvetica", "", 8.8)
        self.set_text_color(51, 65, 85)
        self.multi_cell(self.epw, 4.2, self._clean_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def render_table(self, col1_header, col2_header, rows, col1_width=52):
        col2_width = self.epw - col1_width
        if self.get_y() > self.h - 40:
            self.add_page()

        # Header Row
        self.set_fill_color(241, 245, 249)
        self.set_draw_color(203, 213, 225)
        self.set_line_width(0.2)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(30, 41, 59)
        self.cell(col1_width, 6, f"  {col1_header}", border=1, fill=True)
        self.cell(col2_width, 6, f"  {col2_header}", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Body Rows
        fill = False
        for c1, c2 in rows:
            clean_c1 = self._clean_text(c1)
            clean_c2 = self._clean_text(c2)
            
            # Estimate row height based on c2 text length
            c2_lines = len(clean_c2) // int(col2_width * 0.42) + clean_c2.count('\n') + 1
            row_h = max(5.5, c2_lines * 3.8 + 2)

            if self.get_y() + row_h > self.h - 18:
                self.add_page()
                # Re-print header on new page
                self.set_fill_color(241, 245, 249)
                self.set_font("Helvetica", "B", 8.5)
                self.set_text_color(30, 41, 59)
                self.cell(col1_width, 6, f"  {col1_header}", border=1, fill=True)
                self.cell(col2_width, 6, f"  {col2_header}", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            y_start = self.get_y()
            x_start = self.get_x()

            bg_fill = (248, 250, 252) if fill else (255, 255, 255)
            self.set_fill_color(*bg_fill)

            # Draw background rectangle
            self.rect(x_start, y_start, self.epw, row_h, style="F")

            # Draw Col 1 (code / file font)
            self.set_xy(x_start + 1.5, y_start + 1)
            self.set_font("Courier", "B", 8)
            self.set_text_color(30, 58, 138)
            self.multi_cell(col1_width - 3, 3.8, clean_c1)

            # Draw Col 2 (regular text)
            self.set_xy(x_start + col1_width + 1.5, y_start + 1)
            self.set_font("Helvetica", "", 8.2)
            self.set_text_color(51, 65, 85)
            self.multi_cell(col2_width - 3, 3.8, clean_c2)

            # Draw cell borders
            self.rect(x_start, y_start, col1_width, row_h)
            self.rect(x_start + col1_width, y_start, col2_width, row_h)

            self.set_xy(x_start, y_start + row_h)
            fill = not fill
        self.ln(3)

    def _clean_text(self, text):
        """Sanitize unicode to latin-1 safe characters for Helvetica."""
        if not text:
            return ""
        replacements = {
            "—": "--",
            "–": "-",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "•": "-",
            "→": "->",
            "≥": ">=",
            "≤": "<=",
            "×": "x",
            "≈": "~",
            "…": "...",
            "\u200b": "",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_revision_sheet(output_path):
    pdf = RevisionSheetPDF()
    pdf.add_page()

    # --- Title Block ---
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_line_width(0.4)
    pdf.rect(pdf.l_margin, pdf.get_y(), pdf.epw, 22, style="FD")
    pdf.set_xy(pdf.l_margin + 4, pdf.get_y() + 3)

    pdf.set_font("Helvetica", "B", 13.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "Interview Revision Sheet (v2)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.set_x(pdf.l_margin + 4)
    pdf.cell(0, 5, "Adaptive Agent Orchestration Framework for Autonomous Data Analysis", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(100, 116, 139)
    pdf.set_x(pdf.l_margin + 4)
    pdf.cell(0, 4.5, "Pure Q&A format for rapid interview prep -- read top to bottom before your technical interview", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # --- SECTION A ---
    pdf.section_banner("SECTION A: Core Workflow & Agents")

    pdf.qa_pair(
        "Q: What is this project, in one sentence?",
        "A: Instead of always using multiple AI agents for every task -- which is wasteful for simple requests -- I built a system that measures how complex a task actually is and only uses multiple agents when the task genuinely needs it. Simple tasks get one lightweight agent; complex tasks get a full multi-agent chain with statistical validation."
    )

    pdf.qa_pair(
        "Q: Walk me through what happens step by step when a user submits a request.",
        "A:\n"
        "1. User uploads CSV file(s) and types a question in plain English.\n"
        "2. The system profiles the dataset -- pure pandas, no AI -- counting rows, columns, missing values, and duplicates.\n"
        "3. The user's question is sent to Gemini ONE time and converted into a structured checklist of operations, e.g. ['clean_missing_values', 'group_comparison', 'bar_chart'].\n"
        "4. A complexity score (0-10) is calculated using a rule-based weighted formula based on that checklist and the dataset profile.\n"
        "5. Based on the score, the system routes to one of two paths:\n"
        "   - Simple/Moderate -> General Agent only\n"
        "   - Complex -> Data Prep Agent -> Analysis Agent -> Insight/Validation Agent\n"
        "6. A final report is generated -- mostly templated from real computed data, with one more Gemini call for the executive summary paragraph.\n"
        "7. The user gets: charts, stats, insights, a full execution trace, and a downloadable PDF report with the charts embedded."
    )

    pdf.qa_pair(
        "Q: Does the user's original query get passed to every single agent?",
        "A: No. The query is read by Gemini only ONCE, at the very start, and converted into a structured checklist. Every downstream agent works off that checklist, not the original sentence. This is more efficient (fewer LLM calls) and more consistent (no risk of different agents interpreting the same sentence differently). The only two exceptions: the Insight Agent looks at the original query again alongside the computed numbers so its findings stay relevant, and the Report Generator shows the original question at the top of the report and uses it again for the executive summary."
    )

    pdf.qa_pair(
        "Q: How many agents are there, and what does each one do?",
        "A: Six pieces total, though not all six run on every request:\n"
        "1. Orchestrator -- profiles data, parses the query, scores complexity, routes\n"
        "2. General Agent -- handles Simple AND Moderate paths (same agent, both cases)\n"
        "3. Data Prep Agent -- Complex path only; cleans and merges data\n"
        "4. Analysis Agent -- Complex path only; runs stats, correlation, comparisons\n"
        "5. Insight & Validation Agent -- Complex path only; generates findings and statistically verifies them\n"
        "6. Report Generator -- runs for every path; assembles the final report"
    )

    pdf.qa_pair(
        "Q: Does the Complex path use the General Agent too?",
        "A: No. This is a fork, not addition. Simple/Moderate uses ONLY the General Agent. Complex uses ONLY the three-agent chain (Data Prep, Analysis, Insight/Validation). The General Agent never runs on a Complex-path request."
    )

    pdf.qa_pair(
        "Q: What's the actual difference between Simple and Moderate, if they use the same agent?",
        "A: There's no different code path -- it's the SAME General Agent function. The only difference is how many tools get triggered. A Simple task might only call one function (e.g., just descriptive stats). A Moderate task might trigger cleaning + correlation + multiple charts, because more operations were detected in the query. Same agent, different amount of work requested."
    )

    pdf.qa_pair(
        "Q: How does an agent create charts without calling an LLM?",
        "A: Two parts, both fully mechanical, no AI needed:\n"
        "1. Deciding WHICH chart type: a rule table checks the data types of the columns -- e.g. 'date column + numeric column -> line chart', 'category column + numeric column -> bar chart'. Plain if/else logic.\n"
        "2. Actually DRAWING it: matplotlib takes the real numbers and calculates bar heights/positions mathematically, then saves it as an image. Pure geometry and math from numbers, no language understanding involved."
    )

    # --- SECTION B ---
    pdf.section_banner("SECTION B: What's AI vs What's Not")

    pdf.qa_pair(
        "Q: Out of all the steps in the pipeline, which ones actually call the LLM and which ones are rule-based?",
        "A: Roughly 13 total possible steps, only 2-4 of them ever call Gemini:\n"
        "- ALWAYS an LLM call: query parsing (start of every request), report executive summary (end of every request)\n"
        "- SOMETIMES an LLM call: the complexity tiebreaker (only if the score is in an ambiguous zone, ~20% of cases), Data Prep's imputation strategy choice (Complex path only), Insight generation (Complex path only)\n"
        "- NEVER an LLM call: dataset profiling, the complexity scoring formula itself, all chart generation, all descriptive statistics, all statistical validation tests, the retry decision logic, and the report's structural assembly (only the one summary paragraph uses AI)"
    )

    pdf.qa_pair(
        "Q: Why mix rule-based logic and LLM calls instead of using the LLM for everything, or a pure rule-based system?",
        "A: A pure rule-based system can't understand open-ended natural language requests or judge genuinely ambiguous cases. A pure LLM system would be slow, expensive, and non-deterministic for things that don't need judgment at all -- like deciding a bar chart fits categorical-vs-numeric data, which is just a lookup, not a decision requiring intelligence. I used the LLM only where language understanding or genuine judgment was needed, and rules for everything mechanical or formulaic. This keeps cost and latency down while still handling natural language input."
    )

    pdf.qa_pair(
        "Q: Why not use a learning-based (ML-trained) approach to decide routing, instead of a hand-written formula?",
        "A: That would need a labeled training dataset of 'task -> correct architecture' pairs, which doesn't exist and would be expensive to create credibly for a student project. A learned router without real training data would either be fabricated or non-functional. The hybrid rule + LLM tiebreaker approach is fully explainable and testable without needing that data, which mattered given the project's core thesis is about explainable decisions."
    )

    # --- SECTION C ---
    pdf.section_banner("SECTION C: Statistical Validation")

    pdf.qa_pair(
        "Q: What is statistical validation, and why does the project need it?",
        "A: The Insight Agent uses an LLM to generate claims like 'North spends more than South.' But an LLM can sound confident even when a claim isn't actually backed by the data. Statistical validation is a math-based check -- not an AI check -- that verifies whether the claim is real:\n"
        "1. First, check sample size: need at least 30 data points, or the claim is immediately marked unsupported due to insufficient data.\n"
        "2. If there's enough data, run an actual test:\n"
        "   - 2 groups being compared -> t-test\n"
        "   - 3+ groups being compared -> ANOVA test (Analysis of Variance)\n"
        "   - two numeric variables relating -> correlation significance test\n"
        "3. These produce a p-value -- roughly, 'how likely is this pattern due to random chance.' Below 5% -> supported. Above that -> unsupported."
    )

    pdf.qa_pair(
        "Q: If a claim fails validation, does the user still get their report?",
        "A: Yes, always. Validation never blocks or withholds the final output. It only adds an honest label to that ONE claim -- 'supported' or 'unsupported (with caveat)'. Every chart, stat, and the rest of the report are delivered regardless. Design philosophy: show the user everything, but be transparent about what's statistically confirmed versus just an observation."
    )

    pdf.qa_pair(
        "Q: What happens specifically when a claim is unsupported?",
        "A: One bounded retry is allowed. If unsupported and no retry has happened yet, feedback goes back to the Analysis Agent ('try segmenting differently') and validation is re-attempted once. If still unsupported after that, the system permanently stops retrying and reports the finding with a caveat. Confirmed through direct testing that retry count never exceeds 1."
    )

    # --- SECTION D ---
    pdf.section_banner("SECTION D: LangGraph")

    pdf.qa_pair(
        "Q: What is LangGraph, and how is it used in this project? (simple version)",
        "A: Think of it like a conveyor belt with a shared clipboard. Each agent is a station on the belt. Work moves station to station, and everyone writes their results onto the same shared clipboard so later stations can see what earlier stations found. LangGraph decides which station the work goes to next, and allows one special station (Insight/Validation) to send the work backward to an earlier station (Analysis) for exactly one retry."
    )

    pdf.qa_pair(
        "Q: How do I explain LangGraph using proper technical terms?",
        "A: Each agent is implemented as a NODE in a LangGraph StateGraph -- a function that takes the current STATE and returns an updated state. All nodes share one state object (AgentState) that flows through the graph. Nodes are connected by EDGES: fixed edges always go the same direction; CONDITIONAL EDGES call a routing function I wrote that inspects the state (like the complexity score) and decides the next node. I have one CYCLE -- Insight/Validation can route back to Analysis once, enabling the bounded retry. The graph is defined once, compiled with graph.compile(), and run per-request with graph.invoke(initial_state)."
    )

    pdf.qa_pair(
        "Q: Isn't LangGraph just a wrapper that sends input to an LLM and gets output back?",
        "A: No. LangGraph doesn't know or care whether a node calls an LLM inside it. It's a workflow control layer managing which function runs next and passing shared state between them. Several of my nodes never call an LLM at all (pure pandas functions) -- LangGraph still orchestrates them the same way."
    )

    pdf.qa_pair(
        "Q: What alternatives to LangGraph exist, and why did you pick it?",
        "A: CrewAI (role-based agents collaborating more freeform, less explicit routing control), Microsoft AutoGen (conversational agents looping until done, suited to open-ended dialogue), LlamaIndex Workflows (a fairly similar event-driven graph alternative). I could also hand-code routing with if/else and a while loop. I chose LangGraph because it made state management and conditional routing explicit and testable -- important since the routing decision IS the core thesis of the project."
    )

    # --- SECTION E ---
    pdf.section_banner("SECTION E: Database")

    pdf.qa_pair(
        "Q: What database did you use, and why?",
        "A: SQLite -- a lightweight, file-based database that needs no separate server process. I chose it deliberately over Postgres because this project runs as a single-user demo, not a production multi-user system, so the extra setup complexity of a real database server would have been unnecessary overhead for what the project needed to prove."
    )

    pdf.qa_pair(
        "Q: What is the database actually used for in this project?",
        "A: Persisting everything needed for explainability and evaluation, since that data needs to survive after a single request finishes (and be queryable later, e.g. for the Streamlit UI or the evaluation harness):\n"
        "- Every agent decision and reasoning (the execution trace)\n"
        "- Every data-cleaning action taken\n"
        "- Every insight generated and whether it was statistically validated\n"
        "- Session-level metadata (the query, complexity score, routing decision)\n"
        "- Evaluation run metrics (execution time, LLM call count, cost estimate)\n"
        "Without persistence, all of this would only exist in memory during one request and vanish afterward -- the database is what makes the Execution Trace tab and the evaluation comparison table possible."
    )

    pdf.qa_pair(
        "Q: What tables does the database have?",
        "A: Five tables, one SQLite file:\n"
        "1. sessions -- one row per analysis request: the query, dataset names, complexity score/level, routing reason\n"
        "2. execution_trace -- one row per agent action: which agent, what it did, why, whether it used an LLM call, how long it took\n"
        "3. cleaning_log -- one row per cleaning decision: which column, what issue, what strategy was applied, why\n"
        "4. insights -- one row per generated claim: the evidence, which statistical test was used, the p-value, whether it was supported\n"
        "5. eval_runs -- one row per evaluation harness run: which architecture mode, execution time, LLM calls, estimated cost, whether it completed"
    )

    pdf.qa_pair(
        "Q: How do these tables relate to each other?",
        "A: Every table (except sessions itself) has a session_id column that ties its rows back to one specific request in the sessions table -- similar to a foreign key relationship, though I didn't enforce strict foreign key constraints in SQLite for simplicity. session_id acts like a request ID: given one session_id, I can pull every trace step, every cleaning action, and every insight that happened during that one specific run."
    )

    pdf.qa_pair(
        "Q: Basic database concepts you should be comfortable explaining if asked:",
        "A:\n"
        "- Primary key: a column that uniquely identifies each row (e.g., session_id in sessions, or an auto-incrementing trace_id in execution_trace)\n"
        "- Auto-increment: SQLite automatically assigns the next number for columns like trace_id, so I don't have to generate IDs manually\n"
        "- Why not one big table instead of five: separating concerns -- trace data, cleaning data, and insights are logically different kinds of records with different fields, and keeping them separate makes querying each one simpler and avoids a table with many mostly-empty columns"
    )

    # --- SECTION F ---
    pdf.section_banner("SECTION F: Design & Scope Decisions")

    pdf.qa_pair(
        "Q: Your original plan had 4 complexity levels and 2 feedback loops. Why did you cut it down to 3 levels and 1 loop?",
        "A: The 4th level ('High-Risk') was really just 'Complex, but validation triggers a retry' -- not a genuinely different architecture, so it collapsed naturally into Complex. Two independent feedback loops (cleaning-to-analysis AND insight-to-validation) made the system much harder to guarantee termination for and harder to demo reliably. One bounded loop keeps the system's behavior provable and testable, which mattered more than covering every theoretically possible scenario."
    )

    pdf.qa_pair(
        "Q: Why evaluate with a small case study instead of a formal research experiment?",
        "A: Rigor at the level of a formal controlled experiment wasn't achievable or necessary at student-project scope -- I don't have the resources for statistically significant sample sizes. Instead, I built a fixed set of 6 representative test cases and measured real metrics (time, LLM calls, cost, routing accuracy) across single/multi/adaptive modes. It's framed honestly as a demonstration, not peer-reviewed research."
    )

    pdf.qa_pair(
        "Q: Why run the backend and frontend as two separate applications (FastAPI + Streamlit) instead of one?",
        "A: Separation of concerns -- FastAPI handles the actual pipeline logic and could be reused by any frontend or even called directly via API. Streamlit is purely for the interactive UI layer. This also mirrors how real production systems are typically structured, with an API layer independent of any particular UI."
    )

    pdf.qa_pair(
        "Q: Why does the pipeline run synchronously (the user waits) instead of in the background?",
        "A: A deliberate scope decision. Since this runs as a single-user demo, not a production system with concurrent users, an async task queue would have been unnecessary infrastructure complexity. In a real production version, I'd add one specifically to avoid blocking the user for 30-90 seconds per request."
    )

    pdf.qa_pair(
        "Q: What is the 'shared state' pattern, and why use it instead of just passing variables between functions?",
        "A: All agents read from and write to one shared object (AgentState, a Python TypedDict) rather than passing individual arguments between functions. This means any agent can see everything earlier agents already computed without needing new parameters threaded through every function signature, and it's exactly what LangGraph uses to move data through the graph between nodes. It also made the execution trace logging simple, since every node can append to the same shared trace list."
    )

    pdf.qa_pair(
        "Q: What edge cases did you specifically handle?",
        "A: Empty datasets (0 rows), single-column datasets, high-cardinality 'ID-like' columns being mistakenly treated as real categories, and queries that don't parse into any recognized operation (falls back to a default instead of returning nothing). Also added input validation at the API layer for empty files, empty queries, and headers-only CSVs, rejecting them clearly with proper error messages instead of failing deep inside the pipeline."
    )

    # --- SECTION G ---
    pdf.section_banner("SECTION G: Testing, Bugs, and Process")

    pdf.qa_pair(
        "Q: What real bugs did you find while building this? Give me a specific example.",
        "A: (Pick 2 of these 5 to tell as short stories -- symptom, root cause, fix, verification)\n\n"
        "1. TOKEN TRUNCATION: A complex query that should have scored ~9/10 instead scored 2.5/10. Root cause: Gemini 3.x spends part of the response budget on internal 'thinking' before answering; the token limit was too low, silently cutting off the JSON response, which broke parsing and fell back to a default. Fixed by raising the token budget and using the correct thinking_level parameter (not thinking_budget, which is for an older model and errors on Gemini 3.x).\n\n"
        "2. NUMPY BOOLEAN CRASH: A real request returned a 500 error -- 'numpy.bool object is not iterable.' Root cause: bool(x) or y only wraps x, not the whole expression, so an unwrapped numpy boolean leaked into the API's JSON response. While fixing it, found a second bug: date detection silently broke on pandas 3.x, which changed the default text column type from 'object' to a dedicated 'str' type.\n\n"
        "3. MODERATE PATH NEVER CLEANED DATA: Uploaded messy data, asked to clean it, got 'No cleaning was required' regardless. Root cause: the General Agent had zero cleaning capability at all, contradicting my own design spec. Fixed by adding lightweight, deterministic cleaning to it.\n\n"
        "4. ID COLUMN POLLUTING CORRELATION: The system generated an insight about 'customer ID correlating with spend' -- meaningless. Fixed by excluding ID-like columns from numeric analysis everywhere.\n\n"
        "5. MULTI-FILE UPLOAD DIDN'T ACTUALLY MERGE: Uploaded 3 regional files, chart only showed ONE region. Root cause: the pipeline only ever analyzed the first uploaded file. Implemented a real merge, with the trace explicitly logging it's a naive union merge, not a key-based join."
    )

    pdf.qa_pair(
        "Q: How did you test this without spending money on API calls constantly?",
        "A: I wrote offline smoke tests for every development phase that monkeypatch (replace) the LLM call functions with deterministic stand-ins, so I could verify the pipeline's logic, routing, and retry behavior without needing a live API key for every test run. Only after the offline logic was verified did I test against the real API to catch real-world issues like the token truncation bug, which the mocked tests couldn't have caught since they don't exercise real model behavior."
    )

    pdf.qa_pair(
        "Q: What's the biggest limitation of this project?",
        "A: Two honest answers:\n"
        "1. Complexity scoring weights were a reasonable starting point, refined using my evaluation harness's mismatches, but not tuned against a large labeled dataset.\n"
        "2. Multi-dataset merging is a naive row-concat union, not a key-based join -- no entity resolution across datasets."
    )

    pdf.qa_pair(
        "Q: How would you scale this to production?",
        "A: Add an async task queue so users aren't blocked waiting per request. Move from SQLite to Postgres for concurrent access. Add retries/circuit breakers around the LLM API calls specifically for resilience, separate from the statistical validation retry logic."
    )

    pdf.qa_pair(
        "Q: Why Gemini instead of GPT or Claude?",
        "A: Originally built and tested against Claude, migrated to Gemini for its free tier to iterate without API cost concerns during development. The migration only touched one client module, since everything else calls a generic generate_text() wrapper -- that abstraction made swapping providers straightforward."
    )

    pdf.qa_pair(
        "Q: What would you say about your development process if asked?",
        "A: I didn't just design this and hope it worked -- I ran it against a real API, hit real errors, traced them to actual root causes, verified every fix with a real re-run, and documented all of it. That's a stronger signal than a system that 'just worked' the first time."
    )

    # --- SECTION H ---
    pdf.section_banner("SECTION H: Code File Map -- 'Where is X in your code?'")
    pdf.body_paragraph("Use this if asked to locate a specific feature. Organized by what each file is responsible for.")

    pdf.subsection_title("1. backend/app/agents/ -- the actual agent implementations")
    pdf.render_table(
        "File", "What's inside",
        [
            ("orchestrator.py", "run_orchestrator() -- profiling, query parsing, complexity scoring, routing decision.\nroute_by_complexity() -- the conditional edge function LangGraph calls to decide Simple/Moderate vs Complex."),
            ("general_agent.py", "run_general_agent() -- handles Simple AND Moderate. Lightweight deterministic cleaning (_clean_missing_values_simple()), stats, chart selection."),
            ("data_prep_agent.py", "run_data_prep_agent() -- Complex path only. Missing value/duplicate/outlier detection, LLM-assisted imputation strategy, and _get_working_dataframe() multi-dataset merge."),
            ("analysis_agent.py", "run_analysis_agent() -- Complex path only. Stats, correlation, group comparison, chart selection. Handles the retry re-run logic (retry_instruction branch)."),
            ("insight_validation_agent.py", "run_insight_validation_agent() -- insight generation (LLM), statistical validation (_validate_group_comparison(), _validate_correlation()), and should_retry() conditional edge."),
        ],
        col1_width=48
    )

    pdf.subsection_title("2. backend/app/core/ -- shared logic, not agent-specific")
    pdf.render_table(
        "File", "What's inside",
        [
            ("profiling.py", "profile_datasets(), profile_single_dataset() -- pure pandas dataset profiling. _looks_like_datetime() -- date-string detection heuristic."),
            ("task_understanding.py", "parse_user_query() -- the one LLM call that converts user sentence into a structured operations list. KNOWN_OPERATIONS -- the fixed vocabulary."),
            ("complexity_scoring.py", "compute_complexity_score() -- the weighted rule-based formula. assess_complexity() -- wraps scoring plus LLM tiebreaker logic. THRESHOLDS boundaries."),
            ("analysis_tools.py", "Deterministic stats & charts: describe_stats(), correlation_matrix(), groupby_compare(), plot_bar/line/histogram/scatter/heatmap(), select_chart_type() rule table."),
            ("tracing.py", "log_trace() -- records agent decisions, writing to in-memory state and the database. StepTimer -- measures how long each step takes."),
            ("state.py", "AgentState -- the shared TypedDict schema that flows through the whole LangGraph graph."),
            ("graph.py", "build_graph() -- defines all nodes, fixed/conditional edges, and retry cycle; wires up LangGraph StateGraph. THE file to open if asked 'show me architecture in code.'"),
            ("report_generator.py", "generate_report() -- assembles markdown report from state, with one LLM call for the executive summary."),
            ("pdf_report.py", "generate_pdf_report() -- builds the downloadable PDF with embedded chart images using fpdf2."),
            ("session_runner.py", "run_session() -- bridges FastAPI layer and LangGraph pipeline; computes evaluation metrics (_compute_metrics()) and handles forced-mode overrides."),
            ("llm_client.py", "generate_text() -- single wrapper every agent calls to talk to Gemini. ONLY file that knows about Gemini SDK directly -- everything else is provider-agnostic."),
            ("eval_cases.py", "The 6 fixed test cases (dataset generators + queries + expected complexity levels) used for evaluation."),
            ("eval_runner.py", "run_all_cases() -- runs every case through single/multi/adaptive modes and builds comparison table."),
        ],
        col1_width=48
    )

    pdf.subsection_title("3. backend/app/api/ -- the web layer")
    pdf.render_table(
        "File", "What's inside",
        [
            ("main.py", "FastAPI app. Endpoints: /analyze (runs full pipeline), /trace/{id}, /report/{id}, /report/{id}/download (PDF), /chart/{id}/{filename}, /health."),
        ],
        col1_width=48
    )

    pdf.subsection_title("4. backend/app/db/ -- persistence")
    pdf.render_table(
        "File", "What's inside",
        [
            ("schema.sql", "The 5 table definitions (sessions, execution_trace, cleaning_log, insights, eval_runs)."),
            ("connection.py", "init_db(), get_connection() -- SQLite connection handling."),
        ],
        col1_width=48
    )

    pdf.subsection_title("5. backend/app/tests_manual/ -- offline verification")
    pdf.render_table(
        "File", "What's inside",
        [
            ("phase1_smoke_test.py .. phase6_edge_case_test.py", "One test file per development phase, with LLM calls mocked so they run without live API key. Good to mention if asked 'how did you test this.'"),
        ],
        col1_width=48
    )

    pdf.subsection_title("6. frontend/ -- Streamlit interactive UI")
    pdf.render_table(
        "File", "What's inside",
        [
            ("app.py", "Entire Streamlit UI -- upload form, and all 7 result tabs (Overview, Data Quality, EDA, Visualizations, Insights, Execution Trace, Report). Talks to backend purely over HTTP."),
        ],
        col1_width=48
    )

    # --- Quick Memorization Anchor ---
    pdf.ln(2)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_line_width(0.3)
    pdf.rect(pdf.l_margin, pdf.get_y(), pdf.epw, 24, style="FD")
    pdf.set_xy(pdf.l_margin + 3, pdf.get_y() + 2)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 4.5, "Quick Memorization Anchor for the File Map:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 8.2)
    pdf.set_text_color(51, 65, 85)
    pdf.set_x(pdf.l_margin + 3)
    pdf.cell(0, 4, "- agents/ = the workers (5 files, one per agent)   |   db/ = the memory (SQLite persistence)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(pdf.l_margin + 3)
    pdf.cell(0, 4, "- core/ = shared brain & tools (profiling, scoring, LLM wrapper, graph wiring)   |   frontend/ = the face (Streamlit)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(pdf.l_margin + 3)
    pdf.cell(0, 4, "- api/main.py = the front door (FastAPI HTTP endpoints)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(pdf.l_margin + 3)
    pdf.set_font("Helvetica", "B", 8.2)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 4, "Where is actual orchestration logic?  app/core/graph.py for wiring,  app/agents/orchestrator.py for routing.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(output_path)
    print(f"Successfully generated PDF: {output_path}")

if __name__ == "__main__":
    out_dir = r"C:\Users\Hemanthkumar\Downloads"
    pdf_filename = "Interview_Revision_Sheet_Adaptive_Agent_Framework.pdf"
    full_path = os.path.join(out_dir, pdf_filename)
    generate_revision_sheet(full_path)
