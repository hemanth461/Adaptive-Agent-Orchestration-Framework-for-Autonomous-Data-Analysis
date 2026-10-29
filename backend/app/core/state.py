"""
Shared Agent State.

The single object threaded through every LangGraph node. See PRD Section 7.
"""
from typing import TypedDict, List, Optional
import pandas as pd


class AgentState(TypedDict, total=False):
    # Input
    session_id: str
    raw_datasets: List[pd.DataFrame]
    dataset_names: List[str]
    user_query: str

    # Evaluation-harness override (Phase 5 / M13). Not used in normal operation.
    _forced_mode: Optional[str]
    adaptive_recommended_level: Optional[str]

    # Profiling & routing
    dataset_profile: dict
    task_operations: List[str]
    complexity_score: float
    complexity_breakdown: dict
    complexity_level: str          # "simple" | "moderate" | "complex"
    used_llm_tiebreaker: bool
    routing_reason: str

    # Working data
    cleaned_data: List[pd.DataFrame]
    cleaning_log: List[dict]
    cleaned_csv_path: Optional[str]
    cleaned_csv_filename: Optional[str]

    analysis_results: dict
    visualizations: List[dict]

    insights: List[dict]
    validation_results: List[dict]
    retry_count: int
    retry_instruction: Optional[str]

    # Traceability
    execution_trace: List[dict]
    final_report: Optional[str]
    report_pdf_path: Optional[str]
