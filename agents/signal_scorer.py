"""
SIGNAL_SCORER node: computes behavioral signal scores for top candidates.
"""

from __future__ import annotations
from models.schemas import PipelineState, CandidateScore
from utils.scoring import signal_score


def signal_scorer_node(state: PipelineState) -> PipelineState:
    """LangGraph node: score each candidate's behavioral signals."""
    print(f"\n[SIGNAL_SCORER] Scoring behavioral signals for {len(state.top_candidate_ids)} candidates…")

    # Build quick lookup: candidate_id → Candidate
    cand_map = {c.candidate_id: c for c in state.candidates}

    semantic_scores = getattr(state, "_semantic_scores", {})
    score_map: dict[str, CandidateScore] = {}

    for cid in state.top_candidate_ids:
        candidate = cand_map.get(cid)
        if candidate is None:
            continue
        sig_s = signal_score(candidate)
        score_map[cid] = CandidateScore(
            candidate_id=cid,
            semantic_score=semantic_scores.get(cid, 0.0),
            signal_score=sig_s,
        )

    state.scores = list(score_map.values())
    print(f"[SIGNAL_SCORER] Scored {len(state.scores)} candidates")
    return state
