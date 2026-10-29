"""
Graph Construction.

Wires the full pipeline into a LangGraph StateGraph:
    orchestrator -> [single_agent_path | complex_path]

complex_path = data_prep_agent -> analysis_agent -> insight_validation_agent
                                        ^                    |
                                        |    (bounded retry, max 1)
                                        +--------------------+
"""
import os
import pandas as pd
from langgraph.graph import StateGraph, END

from app.core.state import AgentState
from app.agents.orchestrator import run_orchestrator, route_by_complexity
from app.agents.general_agent import run_general_agent
from app.agents.data_prep_agent import run_data_prep_agent
from app.agents.analysis_agent import run_analysis_agent
from app.agents.insight_validation_agent import run_insight_validation_agent, should_retry
from app.core.report_generator import generate_report
from app.core.pdf_report import generate_pdf_report


def _report_node(state: dict) -> dict:
    session_id = state.get("session_id", "")
    cleaned = state.get("cleaned_data")
    if cleaned and len(cleaned) > 0 and isinstance(cleaned[0], pd.DataFrame):
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "outputs", session_id)
        os.makedirs(out_dir, exist_ok=True)
        csv_filename = "cleaned_dataset.csv"
        csv_path = os.path.join(out_dir, csv_filename)
        cleaned[0].to_csv(csv_path, index=False)
        state["cleaned_csv_path"] = csv_path
        state["cleaned_csv_filename"] = csv_filename

    try:
        generate_report(state)
    except Exception as e:
        state["final_report"] = f"# Data Analysis Report\n\nCould not generate full report text: {e}"

    try:
        pdf_path = generate_pdf_report(state)
        state["report_pdf_path"] = pdf_path
    except Exception as e:
        state["report_pdf_path"] = None
    return state


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", run_orchestrator)
    graph.add_node("single_agent_path", run_general_agent)

    graph.add_node("data_prep_agent", run_data_prep_agent)
    graph.add_node("analysis_agent", run_analysis_agent)
    graph.add_node("insight_validation_agent", run_insight_validation_agent)
    graph.add_node("report_generator", _report_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_by_complexity,
        {
            "single_agent_path": "single_agent_path",
            "complex_path": "data_prep_agent",
        },
    )

    graph.add_edge("single_agent_path", "report_generator")

    # Complex path chain
    graph.add_edge("data_prep_agent", "analysis_agent")
    graph.add_edge("analysis_agent", "insight_validation_agent")

    # Bounded retry loop: insight_validation_agent -> analysis_agent (once) -> insight_validation_agent -> report
    graph.add_conditional_edges(
        "insight_validation_agent",
        should_retry,
        {
            "retry": "analysis_agent",
            "proceed": "report_generator",
        },
    )

    graph.add_edge("report_generator", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

