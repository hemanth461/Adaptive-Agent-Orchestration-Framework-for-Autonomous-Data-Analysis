# Adaptive Agent Orchestration Framework for Autonomous Data Analysis

An AI-powered data analysis system that **dynamically selects its own agent
architecture** — single agent, tool-augmented single agent, or a coordinated
multi-agent workflow — based on an explainable, logged complexity assessment
of the dataset and the user's request.

Built as a B.Tech AI & Data Science portfolio project. See
`PRD_Adaptive_Agent_Orchestration_Framework.md` for the full design
rationale, scope decisions, and evaluation methodology.

## Core idea

> One agent can handle simple tasks efficiently. Complex tasks — multiple
> datasets, messy data, multiple analytical objectives, statistical
> validation — benefit from specialized agents. This system decides which
> level of collaboration a given task actually needs, and logs exactly why.

## Architecture

```
USER
 │
 ▼
UPLOAD DATASET(S) + ANALYSIS REQUEST      (Streamlit)
 │
 ▼
DATASET PROFILING                          (pandas, no LLM)
 │
 ▼
TASK UNDERSTANDING                         (1 LLM call → structured operations)
 │
 ▼
COMPLEXITY ASSESSMENT                      (rule-based score + conditional LLM tiebreaker)
 │
 ▼
ADAPTIVE ORCHESTRATOR  ── logs routing decision + reason
 │
 ├── SIMPLE / MODERATE  →  General Agent (+ tools)
 │
 └── COMPLEX  →  Data Prep Agent → Analysis Agent → Insight/Validation Agent
                        ▲                                    │
                        └──────── 1 bounded retry ───────────┘
 │
 ▼
REPORT GENERATOR  (templated + 1 LLM call for executive summary)
 │
 ▼
FINAL OUTPUT + EXECUTION TRACE
```

**Stack:** FastAPI · LangGraph · Pandas/NumPy · SciPy/Statsmodels ·
Matplotlib · Streamlit · SQLite · Anthropic API (Claude)

## Project structure

```
backend/
  app/
    agents/               orchestrator, general_agent, data_prep_agent,
                           analysis_agent, insight_validation_agent
    core/                  profiling, task_understanding, complexity_scoring,
                           analysis_tools, tracing, state, graph,
                           report_generator, session_runner,
                           eval_cases, eval_runner
    api/                   FastAPI app (main.py)
    db/                    schema.sql, connection.py
    tests_manual/          offline smoke tests, one per phase (no API key needed)
frontend/
  app.py                   Streamlit UI (7 tabs)
data/
  app.db                   SQLite (created on first run)
  outputs/                 generated chart PNGs, per session_id
```

## Setup

```bash
cd backend
pip install -r requirements.txt --break-system-packages

export ANTHROPIC_API_KEY=sk-...

# initialize the database
python -m app.db.connection

# run the backend
uvicorn app.api.main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
pip install streamlit requests --break-system-packages
streamlit run app.py
```

Then open the Streamlit URL, upload a CSV, and describe what you want
analyzed.

## Verifying without an API key

Every phase has an offline smoke test with LLM calls mocked, so the pipeline
mechanics can be verified without live API access:

```bash
cd backend
python -m app.tests_manual.phase1_smoke_test    # profiling + complexity scoring
python -m app.tests_manual.phase2_smoke_test    # orchestrator + single-agent path
python -m app.tests_manual.phase3_smoke_test    # multi-agent path + bounded retry
python -m app.tests_manual.phase4_smoke_test    # report generation
python -m app.tests_manual.phase5_smoke_test    # evaluation harness (single/multi/adaptive)
python -m app.tests_manual.phase6_edge_case_test  # empty data, single-column, high-cardinality
```

All six pass as of the current build.

## Running the real evaluation (requires API key)

```bash
cd backend
export ANTHROPIC_API_KEY=sk-...
python -m app.core.eval_runner
```

Produces `eval_results.csv` plus a printed comparison table (execution time,
LLM call count, estimated cost, completion rate) across single/multi/adaptive
modes, and an adaptive-routing accuracy report against human-judged expected
levels for the 6 fixed test cases in `eval_cases.py`.

## Known simplifications (documented, not hidden)

- **Group-comparison statistical validation** approximates significance from
  aggregate group means/counts (since the Insight Agent only receives
  summarized `analysis_results`, not raw per-row data) rather than running a
  true t-test against raw values.
- **Token/cost estimates** in the evaluation harness use a flat per-call
  average, not exact API usage accounting.
- **Multi-dataset merging** uses a naive join, not full schema reconciliation.
- **Complexity scoring weights** are a reasonable starting heuristic, not
  empirically tuned against real (non-mocked) LLM behavior yet — see the
  Phase 5 finding below.

## Verified findings (from mocked offline runs — useful project-report material)

- **Adaptive routing matched human-judged expected complexity in 3/6 (50%)**
  test cases. Both misses under-classified genuinely complex tasks as
  "moderate," consistent with the Phase 1 finding that `dataset_complexity`
  and `tool_complexity` weights are currently too conservative. Re-tuning
  against real (non-mocked) results is the natural next step once API access
  is available for a full evaluation run.
- **Multi-agent mode cost ~1.9x more LLM calls on average** than single/
  adaptive mode across the fixed test set (3.83 vs 2.0 avg calls per case),
  directly supporting the project's core cost/overhead thesis.
- **The bounded retry loop was verified to terminate correctly** in two
  explicit test scenarios: a genuinely supported claim (zero retries) and a
  deliberately unsupported claim (exactly one retry, then stop with a
  caveat) — `retry_count` never exceeds 1 in any test run.
- **Two real bugs were found and fixed** during development (not just
  designed-away in theory): a LangGraph state-schema omission that silently
  disabled the evaluation harness's forced-mode override, and a bitwise-NOT-
  on-object-dtype bug in the accuracy report. Both are documented in the
  phase-by-phase notes below for interview reference.

## Interview talking point

> "One agent can perform the complete workflow, and for simple tasks it's
> more efficient. But complex data-analysis tasks involve multiple datasets,
> preprocessing challenges, multiple objectives, and validation requirements
> — using multiple agents for every task adds unnecessary overhead, while
> forcing one agent through every complex task reduces specialization and
> reliability. This system adaptively selects the appropriate level of agent
> collaboration based on a transparent, logged complexity assessment — and
> the evaluation harness shows both the cost savings and the current
> limitations of that assessment honestly."

---

## Phase-by-phase build notes

### Phase 0/1 — Skeleton, Profiling, Complexity Scoring
- Project structure, SQLite schema, `profiling.py` (M1), `task_understanding.py`
  (M2), `complexity_scoring.py` (M3), `tracing.py` (M9).
- Finding: initial scoring weights under-classified a designed "moderate"
  case as "simple" — flagged early for later tuning.

### Phase 2 — Orchestrator + Single-Agent Path
- `orchestrator.py` (M4), `general_agent.py` (M5), `analysis_tools.py`,
  `graph.py` (LangGraph wiring).
- Verified end-to-end: profiling → task parsing → scoring → routing → chart
  generation → execution trace, persisted to both memory and SQLite.

### Phase 3 — Multi-Agent Complex Path
- `data_prep_agent.py` (M6), `analysis_agent.py` (M7),
  `insight_validation_agent.py` (M8), bounded retry loop wired into `graph.py`.
- Verified with two scenarios: a genuinely supported claim (0 retries) and a
  deliberately unsupported one (exactly 1 retry, then stop).
- Documented simplification: group-comparison validation approximates
  significance from aggregate evidence rather than raw per-row data.

### Phase 4 — Reporting, API, Frontend
- `report_generator.py` (M10), `session_runner.py` + `api/main.py` (M12),
  `frontend/app.py` (M11, all 7 tabs).
- Bug found & fixed: group-comparison and chart-selection logic was
  auto-picking ID columns (e.g. `customer_id`) as numeric value columns,
  producing meaningless comparisons. Fixed by excluding `*_id`/`id` columns
  from candidate selection; regression-checked against Phase 3.

### Phase 5 — Evaluation Harness
- `eval_cases.py` + `eval_runner.py` (M13); `forced_mode` wired into real
  routing in `orchestrator.py`.
- Bugs found & fixed: (1) `_forced_mode` missing from the `AgentState`
  TypedDict caused LangGraph to silently strip it, making the override inert;
  (2) `~` bitwise-NOT on an object-dtype boolean pandas column crashed the
  mismatch report — fixed with an explicit `== False` comparison.
- Result: adaptive routing matched expected complexity in 3/6 cases (50%);
  multi-mode cost ~1.9x more LLM calls than single/adaptive.

### Phase 6 — Polish & Edge Cases
- Edge cases handled and verified: empty uploads (0 rows/columns), single-
  column datasets, high-cardinality "ID-like" group columns (previously
  would have produced unreadable 100+ row group comparisons), and queries
  that parse to zero recognized operations (now fall back to
  `descriptive_stats` instead of an empty pipeline).
- API-layer validation added: empty files, empty query text, headers-only
  CSVs all rejected with clear 400 errors instead of failing deeper in the
  pipeline.
- Full regression suite (Phases 1-5 smoke tests) re-run and confirmed
  unaffected by the Phase 6 changes.

## Suggested next steps beyond this build

1. Obtain a live `ANTHROPIC_API_KEY` and re-run `eval_runner.py` for real
   (non-mocked) results; use the 3/6 mismatch finding to re-tune the
   `complexity_scoring.py` weights.
2. Record a 2-3 minute demo video: upload a messy multi-column dataset, show
   the Execution Trace tab, and specifically show the retry loop firing once
   and stopping — this is the single most interview-compelling moment to
   have on video.
3. Add a couple more `eval_cases.py` entries covering genuine multi-dataset
   merge scenarios if time allows, since the current 6 cases all use a
   single dataset.

## Live debugging session (real Gemini API, post-migration)

Once running against the real Gemini API (not mocked), three genuine bugs
were found and fixed through actual use, not just testing:

1. **Silent query-parsing truncation**: Gemini 3.x models spend part of
   `max_output_tokens` on internal "thinking" before the visible response.
   With the original 500-token limit, the JSON response was cut off
   mid-string, breaking parsing and silently falling back to a single
   default operation. Fixed by raising every LLM call site's token budget to
   2048-3000.
2. **ID columns polluting correlation analysis**: `correlation_matrix()` and
   `plot_heatmap()` included identifier columns (e.g. `customer_id`) in
   their numeric column set, producing meaningless correlations that the
   real LLM then generated insights about. Fixed by excluding `*_id`/`id`
   columns, consistent with the exclusion already applied to chart selection
   and group comparison in Phase 4.
3. **Moderate path never cleaned data regardless of what was requested**:
   the General Agent (Simple/Moderate) had no cleaning capability at all,
   contradicting the PRD's own Level 2 example ("Clean minor missing values,
   analyze correlations, and create multiple charts"). Fixed by adding
   lightweight, deterministic (no LLM call) missing-value imputation and
   duplicate removal to the General Agent, triggered by
   `clean_missing_values`/`clean_duplicates` operations.

A fourth item was implemented as a genuine feature addition rather than a
bug fix: **real multi-dataset merging**. Previously, uploading multiple
files only affected the complexity score -- the actual analysis silently
ran on the first uploaded file only. The Data Prep Agent now performs a
naive union merge (row concat, aligned by column name) when
`merge_datasets` is requested and multiple datasets are uploaded, with an
explicit trace log noting this is not a key-based join. Verified with a
3-region customer dataset: previously produced a single-region bar chart
(East only, since it uploaded first); after the fix, all 3 regions appear
correctly in the merged 140-row dataset.

## Two more bugs found via the real /analyze endpoint (not caught by unit-level smoke tests)

7. **numpy.bool_ leaking into the API response, crashing FastAPI's JSON
   encoder**: `is_datetime` in `profiling.py` was computed as
   `bool(...) or _looks_like_datetime(...)` -- when the first operand was
   False, the expression's result was whatever `_looks_like_datetime()`
   returned, which was an unwrapped `numpy.bool_` from a pandas comparison,
   not a plain Python `bool`. FastAPI's `jsonable_encoder` cannot serialize
   numpy scalar types and raised `TypeError("'numpy.bool' object is not
   iterable")`. This only surfaced when a real request produced a
   `dataset_profile` field and went through actual HTTP JSON encoding --
   none of the offline smoke tests exercise that serialization path, which
   is exactly why it wasn't caught until real end-to-end use. Fixed by
   wrapping the entire expression in `bool(...)`.

8. **`is_object_dtype` no longer detects string columns on pandas 3.x**:
   while fixing bug #7, testing revealed `_looks_like_datetime()` was
   silently returning `False` for genuine date-string columns. Root cause:
   pandas 3.0 changed the default dtype for text columns from `object` to a
   dedicated `str` dtype, and `is_object_dtype()` -- written against
   pandas 2.x behavior -- no longer matches. Since `requirements.txt` was
   loosened to `pandas>=2.2.3` earlier (to fix a Windows wheel-build issue),
   pip installed pandas 3.0.2, silently changing this behavior without any
   error. Fixed by switching to `is_string_dtype()`, which correctly covers
   both the legacy `object` dtype and the new pandas 3.x `str` dtype.

Both bugs are good illustrations of a general lesson: unit-level tests with
mocked LLM calls verify pipeline *logic*, but don't exercise real HTTP
serialization or catch library-version-driven behavior changes. Both only
surfaced once the app was actually run end-to-end against a live server
with real data.

## New feature: downloadable PDF report with embedded charts

The markdown report in the Streamlit UI only names visualizations and their
selection reason -- it never embedded the actual chart images. Added a real,
complete downloadable deliverable:

- `backend/app/core/pdf_report.py`: builds a multi-page PDF using fpdf2
  (pure-Python, no system dependencies -- installs cleanly on Windows)
  containing every report section AND every generated chart embedded as an
  actual image, not just referenced by name.
- Generated automatically at the end of every pipeline run (`graph.py`'s
  report node), saved to `data/outputs/{session_id}/report.pdf`.
- New endpoint: `GET /report/{session_id}/download` serves the PDF.
- Streamlit's Report tab now has two download buttons: the existing
  markdown, plus "Download Full Report (PDF, with charts)".

Bug found and fixed during verification: fpdf2's `multi_cell()` left the
cursor's x-position at the right margin after each call instead of resetting
to the left margin, causing every second text block to fail with
"Not enough horizontal space to render a single character." Fixed by
explicitly calling `set_x(self.l_margin)` before and after every text block.
Verified by rendering an actual generated PDF to images and visually
confirming a chart displays correctly with its title, selection reasoning,
and page numbering intact.
