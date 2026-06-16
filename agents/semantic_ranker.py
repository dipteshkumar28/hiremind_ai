"""
SEMANTIC_RANKER node: embeds the JD, performs FAISS similarity search,
and returns the top-N candidate IDs.
"""

from __future__ import annotations
import os
import pickle

from utils.faiss_store import load_vector_store, search_similar_candidates
from models.schemas import PipelineState

# How many candidates to retrieve from FAISS before re-ranking
SEMANTIC_TOP_K = 5000   # retrieve top 5K, re-rank down to 100

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "candidates.faiss")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "candidates_ids.pkl")


def semantic_ranker_node(state: PipelineState) -> PipelineState:
    """LangGraph node: semantic search over FAISS index using JD embedding."""
    print(f"\n[SEMANTIC_RANKER] Running FAISS similarity search (top {SEMANTIC_TOP_K})…")

    # Load index
    index, ids = load_vector_store(INDEX_PATH, META_PATH)

    # Build a rich JD query text that captures what the role is actually about
    jd_req = state.job_requirements
    query = f"""
Senior AI Engineer role requiring:
{state.jd_text[:2000]}

Key requirements:
Required skills: {', '.join(jd_req.required_skills)}
Preferred skills: {', '.join(jd_req.preferred_skills)}
Experience: {jd_req.experience_range}
Role: {jd_req.role_summary}
""".strip()

    results = search_similar_candidates(
        query_text=query,
        index=index,
        ids=ids,
        top_k=SEMANTIC_TOP_K,
    )

    # Store (id, score) pairs for SIGNAL/EXPERIENCE scorers to use
    state.top_candidate_ids = [cid for cid, _ in results]

    # Stash semantic scores in a temporary attribute for FINAL_RANKER
    # (we attach them to scores list in the signal/experience nodes)
    state._semantic_scores = {cid: score for cid, score in results}

    print(f"[SEMANTIC_RANKER] Retrieved {len(results)} candidates from FAISS")
    return state
