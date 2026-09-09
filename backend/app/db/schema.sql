-- Adaptive Agent Orchestration Framework
-- SQLite schema (see PRD Section 9)

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_query TEXT,
    dataset_names TEXT,          -- JSON list
    complexity_score REAL,
    complexity_level TEXT,       -- simple | moderate | complex
    routing_reason TEXT,
    forced_mode TEXT             -- NULL | 'single' | 'multi' (used for eval runs)
);

CREATE TABLE IF NOT EXISTS execution_trace (
    trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    step_order INTEGER,
    agent_name TEXT,
    action TEXT,
    reason TEXT,
    tool_used TEXT,
    llm_call BOOLEAN DEFAULT 0,
    duration_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cleaning_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    column_name TEXT,
    issue_type TEXT,             -- missing | duplicate | dtype | outlier
    strategy_applied TEXT,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS insights (
    insight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    claim TEXT,
    evidence TEXT,                -- JSON
    test_used TEXT,
    p_value REAL,
    supported BOOLEAN,
    retry_count INTEGER DEFAULT 0,
    final_status TEXT             -- supported | unsupported_with_caveat
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    architecture_mode TEXT,       -- single | multi | adaptive
    execution_time_sec REAL,
    llm_call_count INTEGER,
    estimated_tokens INTEGER,
    estimated_cost_usd REAL,
    task_completed BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_trace_session ON execution_trace(session_id);
CREATE INDEX IF NOT EXISTS idx_cleaning_session ON cleaning_log(session_id);
CREATE INDEX IF NOT EXISTS idx_insights_session ON insights(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_session ON eval_runs(session_id);
