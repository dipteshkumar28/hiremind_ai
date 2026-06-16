"""
EXPERIENCE_SCORER node: scores career history, years of experience,
and role relevance for each shortlisted candidate.
"""

from __future__ import annotations
from models.schemas import PipelineState
from utils.scoring import skill_score, experience_score


def experience_scorer_node(state: PipelineState) -> PipelineState:
    """LangGraph node: add skill and experience scores to existing CandidateScore objects."""
    print(f"\n[EXPERIENCE_SCORER] Scoring skills & experience for {len(state.scores)} candidates…")

    cand_map = {c.candidate_id: c for c in state.candidates}
    job_req = state.job_requirements

    for cs in state.scores:
        candidate = cand_map.get(cs.candidate_id)
        if candidate is None:
            continue
        cs.skill_score = skill_score(candidate, job_req)
        cs.experience_score = experience_score(candidate, job_req)

    print(f"[EXPERIENCE_SCORER] Done.")
    return state
