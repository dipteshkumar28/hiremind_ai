"""
FINAL_RANKER node: combines all component scores using the weighted formula
and produces the top-100 ranked list.

Formula:
  final_score = 0.50 * semantic_score (0-1 → 0-100)
              + 0.25 * skill_score    (0-100)
              + 0.15 * signal_score   (0-100)
              + 0.10 * experience_score (0-100)
"""

from __future__ import annotations
from models.schemas import PipelineState, RankingResult
from utils.scoring import final_score


def final_ranker_node(state: PipelineState) -> PipelineState:
    """LangGraph node: compute composite scores and rank top-100."""
    print(f"\n[FINAL_RANKER] Computing composite scores for {len(state.scores)} candidates…")

    for cs in state.scores:
        cs.final_score = final_score(
            semantic_score=cs.semantic_score,
            sk_score=cs.skill_score,
            sig_score=cs.signal_score,
            exp_score=cs.experience_score,
        )

    # Sort descending by final_score, with candidate_id as tie-break (ascending)
    ranked = sorted(state.scores, key=lambda x: (-x.final_score, x.candidate_id))

    top100 = ranked[:100]
    results = []
    for rank, cs in enumerate(top100, start=1):
        results.append(
            RankingResult(
                candidate_id=cs.candidate_id,
                rank=rank,
                score=round(cs.final_score, 4),
                reasoning="",  # filled by REASONING_AGENT
            )
        )

    state.results = results
    # Keep full scored list for reasoning agent to query
    state._ranked_scores = {cs.candidate_id: cs for cs in ranked[:100]}

    print(f"[FINAL_RANKER] Top candidate: {results[0].candidate_id} (score={results[0].score})")
    print(f"[FINAL_RANKER] #100 candidate: {results[-1].candidate_id} (score={results[-1].score})")
    return state
