"""
Graph Construction.

Wires the full pipeline into a LangGraph StateGraph:
    orchestrator -> [single_agent_path | complex_path]

complex_path = data_prep_agent -> analysis_agent -> insight_validation_agent
                                        ^                    |
                                        |    (bounded retry, max 1)
                                        +--------------------+
"""
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
    generate_report(state)
    pdf_path = generate_pdf_report(state)
    state["report_pdf_path"] = pdf_path
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

