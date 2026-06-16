"""
LangGraph StateGraph workflow for HireMind AI.

Nodes (in execution order):
  1. jd_analyzer       → parse JD, extract requirements
  2. candidate_loader  → load 100K candidates.jsonl
  3. embedding_agent   → embed + FAISS index (cached)
  4. semantic_ranker   → FAISS similarity search → top 5K
  5. signal_scorer     → behavioral signal scoring
  6. experience_scorer → skill + career scoring
  7. final_ranker      → weighted composite + top-100
  8. reasoning_agent   → generate explanations
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from models.schemas import PipelineState
from agents.jd_analyzer import jd_analyzer_node
from agents.candidate_loader import candidate_loader_node
from agents.embedding_agent import embedding_agent_node
from agents.semantic_ranker import semantic_ranker_node
from agents.signal_scorer import signal_scorer_node
from agents.experience_scorer import experience_scorer_node
from agents.final_ranker import final_ranker_node
from agents.reasoning_agent import reasoning_agent_node


def build_workflow() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.
    Returns a compiled graph ready to invoke.
    """

    # LangGraph requires the state to be a dict-like or TypedDict.
    # We use PipelineState (Pydantic) and wrap nodes accordingly.
    # LangGraph v0.2+ supports Pydantic models as state directly.
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("jd_analyzer",       jd_analyzer_node)
    graph.add_node("candidate_loader",  candidate_loader_node)
    graph.add_node("embedding_agent",   embedding_agent_node)
    graph.add_node("semantic_ranker",   semantic_ranker_node)
    graph.add_node("signal_scorer",     signal_scorer_node)
    graph.add_node("experience_scorer", experience_scorer_node)
    graph.add_node("final_ranker",      final_ranker_node)
    graph.add_node("reasoning_agent",   reasoning_agent_node)

    # Define edges (linear pipeline)
    graph.set_entry_point("jd_analyzer")
    graph.add_edge("jd_analyzer",       "candidate_loader")
    graph.add_edge("candidate_loader",  "embedding_agent")
    graph.add_edge("embedding_agent",   "semantic_ranker")
    graph.add_edge("semantic_ranker",   "signal_scorer")
    graph.add_edge("signal_scorer",     "experience_scorer")
    graph.add_edge("experience_scorer", "final_ranker")
    graph.add_edge("final_ranker",      "reasoning_agent")
    graph.add_edge("reasoning_agent",   END)

    return graph.compile()
