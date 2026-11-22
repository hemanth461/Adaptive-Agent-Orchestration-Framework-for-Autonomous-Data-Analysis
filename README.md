<div align="center">

# ⚡ Adaptive Agent Orchestration Framework for Autonomous Data Analysis

An enterprise-ready, autonomous multi-agent platform that **dynamically routes data analysis tasks across single-agent and multi-agent workflows** based on real-time complexity scoring—slashing organizational AI compute and API costs while delivering 100% statistically fact-checked analytical deliverables.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://adaptive-agent-orchestration-framework.onrender.com)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hemanth461/Adaptive-Agent-Orchestration-Framework-for-Autonomous-Data-Analysis)
[![Uptime Robot](https://img.shields.io/badge/UptimeRobot-24%2F7_Active-3BD671?style=for-the-badge&logo=uptimerobot&logoColor=white)](https://adaptive-agent-orchestration-framework.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-FF9900?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.x_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)

[**Explore the Live Demo 🚀**](https://adaptive-agent-orchestration-framework.onrender.com) · [Report Bug](https://github.com/hemanth461/Adaptive-Agent-Orchestration-Framework-for-Autonomous-Data-Analysis/issues) · [Request Feature](https://github.com/hemanth461/Adaptive-Agent-Orchestration-Framework-for-Autonomous-Data-Analysis/issues)

</div>

---

## 📌 Executive Summary & Problem Statement

Most modern AI analytics frameworks suffer from a fundamental cost flaw: they force **every single user query** through an identical, heavy multi-agent workflow. Asking for simple descriptive statistics or plotting a single histogram invokes multiple LLM agents, generating unnecessary latency and burning organizational API budgets. Conversely, forcing complex, dirty multi-dataset tasks through a naive single prompt leads to hallucinations and analysis failures.

The **Adaptive Agent Orchestration Framework** introduces an intelligent, cost-saving architecture:
* **Deterministic 4-Factor Complexity Scoring:** Evaluates dataset dimensions, data missingness, duplicate rates, tool complexity, and analytical goals on a `0–10` scale.
* **Dynamic Architecture Routing:**
  * **Level 1 (Simple, Score $\le 3.5$):** Single General Agent executes lightweight statistical summarization.
  * **Level 2 (Moderate, Score $3.6 - 6.5$):** Tool-Augmented Agent with callable pandas, correlation, and multi-chart tools.
  * **Level 3 (Complex, Score $\ge 6.6$):** Coordinated Multi-Agent Team (**Data Prep $\rightarrow$ Analysis $\rightarrow$ Validation $\rightarrow$ Report**) with bounded self-correction.
* **Business Impact:** Reduces token consumption and LLM API costs by **40%–60%** while ensuring all reported insights are validated through rigorous parametric and non-parametric hypothesis testing.

---

## 📸 Product Walkthrough & Visual Proof

### 1. Interactive Analysis Workspace & Demo Presets
> Users can upload arbitrary CSV datasets or launch instant 1-click test scenarios across Simple, Moderate, and Complex tiers.

<div align="center">
  <img src="docs/screenshots/01_landing_and_presets.png" alt="Workspace and Presets" width="95%" />
</div>

---

### 2. Adaptive Architecture Routing & Autonomous Data Cleaning
> The orchestrator logs its mathematical routing score (e.g., `6.92/10 -> Complex`), activates the required agent tier, and executes comprehensive automated data remediation (standardizing text, fixing negative numbers, clamping IQR outliers, and imputing missing fields).

<div align="center">
  <img src="docs/screenshots/02_adaptive_routing_data_cleaning.png" alt="Adaptive Routing and Data Cleaning" width="95%" />
</div>

---

### 3. Publication-Grade Visual Analytics & Progression Curves
> Dual-pattern visualization engine automatically tailors chart designs: ranked dual-panel comparisons (Top 10 Highest vs. Lowest) and smooth progression curves tracking primary metrics against secondary risk factors and national averages.

<div align="center">
  <img src="docs/screenshots/03_visual_analytics.png" alt="Entity Ranking Bar Chart" width="95%" />
  <br><br>
  <img src="docs/screenshots/03_line_progression_trends.png" alt="Progression and Risk Factor Trends" width="95%" />
</div>

---

### 4. Zero-Hallucination Statistical Fact-Checking
> Every generated insight is cross-examined against empirical statistics using One-Way ANOVA F-tests, two-sample t-tests, and correlation significance ($p < 0.05$) to ensure all findings are factually verified before delivery.

<div align="center">
  <img src="docs/screenshots/04_statistical_fact_checking.png" alt="Statistical Fact-Checking" width="95%" />
</div>

---

### 5. Transparent Execution Trace & Architecture Explainability
> Provides a human-readable audit trail tracking every agent handoff, tool invocation, and execution latency, allowing engineering and compliance teams to verify exactly how decisions were made.

<div align="center">
  <img src="docs/screenshots/05_execution_trace.png" alt="Execution Trace" width="95%" />
</div>

---

## 🏗️ Architecture & Orchestration Flow

```mermaid
flowchart TD
    User([User CSV Upload + Query]) --> Profiling[Dataset Profiling\npandas / deterministic]
    Profiling --> TaskUnderstanding[Task Understanding\n1 LLM Call → Structured Checklist]
    TaskUnderstanding --> Scoring[Complexity Assessment\n4-Factor Deterministic Engine: 0-10]
    
    Scoring --> Router{Adaptive Router}
    
    %% Tier 1 & 2
    Router -->|Score <= 6.5\nSimple / Moderate| GeneralAgent[General Agent\nDirect summarization + tools]
    
    %% Tier 3
    Router -->|Score >= 6.6\nComplex| DataPrep[Data Prep Agent\nText normalization, IQR outlier capping, imputation]
    DataPrep --> Analysis[Analysis Agent\nGroup tests, stats, correlations]
    Analysis --> Validation[Validation Agent\nANOVA, t-test, sample validation]
    
    %% Bounded Retry Loop
    Validation -.->|Unsupported Finding\nMax 1 Bounded Retry| Analysis
    
    GeneralAgent --> Reporting[Report Generator\nTemplated + LLM Executive Summary]
    Validation --> Reporting
    
    Reporting --> Outputs([Cleaned CSV + Executive PDF + Interactive UI])
```

---

## ✨ Core Engineering Highlights

* **Cost-Optimized Orchestration:** Reserving heavy multi-agent workflows only for complex tasks cuts operational expenses by 40%–60% over static multi-agent setups.
* **Deterministic Text & State Normalization:** Normalizes casing collisions, removes trailing whitespace, and standardizes 2-letter state abbreviations (`tn` $\rightarrow$ `TN`, `vt` $\rightarrow$ `VT`) across high-cardinality datasets.
* **Robust Outlier Remediation:** Automatically detects non-positive invalid measurements and flags extreme anomalies outside $3\times$ IQR, imputing robust medians to prevent statistical skew.
* **Bounded Reflection & Self-Correction:** When an analytical claim lacks statistical backing ($p \ge 0.05$ or $n < 30$), the system triggers exactly one targeted re-analysis loop before delivering results with transparent caveats.
* **24/7 Production Availability:** Hosted on Render with **UptimeRobot** heartbeat monitoring to prevent cold starts and ensure instant responsiveness.

---

## 🛠️ Tech Stack

| Domain | Tools & Technologies |
|---|---|
| **Multi-Agent Orchestration** | LangGraph (StateGraph, Conditional Edges), Google Gemini API (`gemini-2.5-flash`) |
| **Backend API** | FastAPI, Uvicorn, Pydantic, Python-Multipart |
| **User Interface** | Streamlit, Requests |
| **Data Processing & Statistics** | Pandas, NumPy, SciPy (One-Way ANOVA, Student's t-test, skewness metrics) |
| **Visualization & Reporting** | Matplotlib, Seaborn, FPDF2 (PDF Report Compilation with embedded charts) |
| **Persistence & Tracing** | SQLite3 (Sessions, Execution Trace, Cleaning Log, Validation Insights) |
| **Deployment & Monitoring** | Docker, Render Cloud Platform, UptimeRobot (24/7 Health Monitoring) |

---

## 🚀 Quickstart & Local Installation

### Prerequisites
* Python 3.10+
* Google Gemini API Key ([Get one free here](https://aistudio.google.com/))

### 1. Clone the Repository
```bash
git clone https://github.com/hemanth461/Adaptive-Agent-Orchestration-Framework-for-Autonomous-Data-Analysis.git
cd Adaptive-Agent-Orchestration-Framework-for-Autonomous-Data-Analysis
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-api-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# Initialize SQLite database
python -m app.db.connection

# Launch FastAPI backend on port 8000
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Frontend Setup (In a separate terminal)
```bash
cd frontend
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Verification & Offline Testing (No API Key Required)

Deterministic offline smoke tests verify the routing, pipeline mechanics, and edge-case handling with mocked LLM calls:

```bash
cd backend
python -m app.tests_manual.phase1_smoke_test      # Profiling & complexity scoring
python -m app.tests_manual.phase2_smoke_test      # Orchestrator & single-agent routing
python -m app.tests_manual.phase3_smoke_test      # Multi-agent chain & bounded retry
python -m app.tests_manual.phase4_smoke_test      # Report compilation & PDF generation
python -m app.tests_manual.phase6_edge_case_test  # 0-row, single-column, & ID-like handling
```

---

## 📁 Repository Structure

```text
adaptive-agent-framework/
├── backend/
│   ├── app/
│   │   ├── agents/            # Orchestrator, General, Data Prep, Analysis, Validation
│   │   ├── api/               # FastAPI endpoints (/analyze, /health, /report/download)
│   │   ├── core/              # Profiling, Complexity Scoring, Analysis Tools, Graph Wiring, PDF
│   │   ├── db/                # SQLite schema and connection manager
│   │   └── tests_manual/      # Phase-by-phase offline smoke tests
│   ├── scripts/               # Utility scripts (PDF revision sheet generator)
│   └── requirements.txt
├── frontend/
│   └── app.py                 # Streamlit UI with 7 analytical tabs & demo presets
├── docs/
│   └── screenshots/           # UI walkthrough screenshots
├── start.sh                   # Dual-process startup script for Render deployment
└── README.md
```

---

## 👤 Author

* **Hemanth Kumar**  
  * GitHub: [@hemanth461](https://github.com/hemanth461)  
  * Live Project: [Adaptive Agent Orchestration Framework](https://adaptive-agent-orchestration-framework.onrender.com)
