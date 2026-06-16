"""
REASONING_AGENT node: generates human-readable explanations for each
of the top-100 candidates WITHOUT making individual LLM calls per candidate
(would be 100 API calls).

Instead it uses a template-based approach with rule-derived bullet points,
and one optional LLM call to summarise any edge cases in the top-10.
"""

from __future__ import annotations
import os
from datetime import date, datetime
from typing import Optional

from models.schemas import (
    PipelineState, RankingResult, Candidate, CandidateScore, JobRequirements
)
from utils.scoring import AI_CORE_SKILLS, CONSULTING_COMPANIES


def _days_since(date_str: str) -> Optional[int]:
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (date.today() - dt).days
    except Exception:
        return None


def _build_reasoning(
    candidate: Candidate,
    cs: CandidateScore,
    job_req: JobRequirements,
) -> str:
    """Rule-based reasoning string (short enough for CSV)."""
    p = candidate.profile
    s = candidate.redrob_signals

    # Positives
    positives = []

    # Title relevance
    ai_titles = {"ai engineer", "ml engineer", "machine learning", "data scientist",
                 "nlp", "research engineer", "applied scientist"}
    if any(t in p.current_title.lower() for t in ai_titles):
        positives.append(f"Title: {p.current_title}")

    # Years
    yoe = p.years_of_experience
    if 5 <= yoe <= 9:
        positives.append(f"{yoe}yrs exp (target 5-9)")

    # Matching skills
    req_lower = {r.lower() for r in job_req.required_skills}
    matched = [sk.name for sk in candidate.skills
               if any(r in sk.name.lower() or sk.name.lower() in r for r in req_lower)
               or any(ai in sk.name.lower() for ai in AI_CORE_SKILLS)]
    if matched:
        positives.append(f"Skills: {', '.join(matched[:5])}")

    # Assessments
    if s.skill_assessment_scores:
        avg = sum(s.skill_assessment_scores.values()) / len(s.skill_assessment_scores)
        positives.append(f"Avg assessment: {avg:.0f}/100")

    # Response rate
    if s.recruiter_response_rate >= 0.5:
        positives.append(f"Response rate: {s.recruiter_response_rate:.0%}")

    # Active
    days = _days_since(s.last_active_date)
    if days is not None and days <= 30:
        positives.append("Active recently")

    # GitHub
    if s.github_activity_score >= 50:
        positives.append(f"GitHub: {s.github_activity_score:.0f}/100")

    # Product company
    all_consulting = all(
        any(c in e.company.lower() for c in CONSULTING_COMPANIES)
        for e in candidate.career_history if e.company
    )
    if not all_consulting:
        positives.append("Product co. background")

    # Open to work
    if s.open_to_work_flag:
        positives.append("Open to work")

    # Negatives / gaps
    negatives = []
    if yoe < 4 or yoe > 12:
        negatives.append(f"Exp out of range ({yoe}yrs)")
    if not s.open_to_work_flag:
        negatives.append("Not marked open-to-work")
    if days is not None and days > 90:
        negatives.append(f"Inactive {days}d")
    if all_consulting:
        negatives.append("Pure consulting background")
    if s.notice_period_days > 60:
        negatives.append(f"Notice: {s.notice_period_days}d")

    # Compose
    pos_str = "; ".join(positives) if positives else "No strong positives"
    neg_str = "; ".join(negatives) if negatives else "No major gaps"
    return (
        f"{p.current_title}, {yoe}yrs exp; "
        f"scores sem={cs.semantic_score:.3f} skl={cs.skill_score:.0f} "
        f"sig={cs.signal_score:.0f} exp={cs.experience_score:.0f}. "
        f"Strengths: {pos_str}. "
        f"Gaps: {neg_str}."
    )


def reasoning_agent_node(state: PipelineState) -> PipelineState:
    """LangGraph node: attach human-readable reasoning to each top-100 result."""
    print(f"\n[REASONING_AGENT] Generating reasoning for {len(state.results)} candidates…")

    cand_map = {c.candidate_id: c for c in state.candidates}
    ranked_scores = getattr(state, "_ranked_scores", {})
    job_req = state.job_requirements

    updated_results = []
    for result in state.results:
        candidate = cand_map.get(result.candidate_id)
        cs = ranked_scores.get(result.candidate_id)
        if candidate and cs:
            result.reasoning = _build_reasoning(candidate, cs, job_req)
        else:
            result.reasoning = f"{result.candidate_id}: insufficient data"
        updated_results.append(result)

    state.results = updated_results
    print(f"[REASONING_AGENT] Done.")
    return state
