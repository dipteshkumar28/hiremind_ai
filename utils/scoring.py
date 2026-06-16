"""
Scoring helpers for HireMind AI.
All functions return values in [0, 100] unless noted.
"""

from __future__ import annotations
import math
from datetime import date, datetime
from typing import List, Dict, Optional

from models.schemas import Candidate, JobRequirements, RedrobSignals


# ─────────────────────────────────────────────
# Skill scoring
# ─────────────────────────────────────────────

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.80,
    "expert": 1.00,
}

# AI/ML core skills as defined by the JD and sample submission signal
AI_CORE_SKILLS = {
    # Embeddings & retrieval
    "embeddings", "vector database", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch", "sentence transformers", "bge", "e5",
    # LLMs & fine-tuning
    "llm", "large language models", "fine-tuning llms", "lora", "qlora", "peft",
    "rag", "retrieval augmented generation", "langchain", "langgraph",
    # ML / ranking
    "machine learning", "deep learning", "nlp", "natural language processing",
    "learning to rank", "information retrieval", "recommendation systems",
    "ranking", "re-ranking", "hybrid search",
    # MLOps / infra
    "mlops", "model serving", "bentoml", "triton", "ray", "kubeflow",
    "weights & biases", "mlflow",
    # Data / Python
    "python", "pytorch", "tensorflow", "scikit-learn", "hugging face",
    "transformers", "xgboost",
}

# Consulting-firm disqualifier (per JD)
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware",
}

# Product-company bonus industries
PRODUCT_COMPANY_INDUSTRIES = {
    "saas", "software", "technology", "fintech", "edtech", "healthtech",
    "e-commerce", "media", "gaming", "ai", "ml",
}


def skill_score(candidate: Candidate, job_req: JobRequirements) -> float:
    """
    Score [0, 100] based on:
    - Match to JD required / preferred skills
    - Proficiency level
    - Endorsements (trust signal)
    - Penalty for pure consulting background
    """
    if not candidate.skills:
        return 0.0

    req_skills_lower = {s.lower() for s in job_req.required_skills}
    pref_skills_lower = {s.lower() for s in job_req.preferred_skills}

    total_weight = 0.0
    matched_required = 0
    matched_preferred = 0

    for sk in candidate.skills:
        name_lower = sk.name.lower()
        prof_w = PROFICIENCY_WEIGHT.get(sk.proficiency, 0.25)
        # Endorsement bonus (log scale, capped)
        endorse_bonus = min(math.log1p(sk.endorsements) / math.log1p(100), 0.3)
        # Duration bonus (more months = more trust)
        duration_bonus = min(sk.duration_months / 60, 0.2)
        effective_w = prof_w * (1 + endorse_bonus + duration_bonus)

        # Direct JD match
        if any(r in name_lower or name_lower in r for r in req_skills_lower):
            total_weight += effective_w * 3.0
            matched_required += 1
        elif any(p in name_lower or name_lower in p for p in pref_skills_lower):
            total_weight += effective_w * 1.5
            matched_preferred += 1
        # AI core skill match
        elif any(ai in name_lower for ai in AI_CORE_SKILLS):
            total_weight += effective_w * 1.0

    # Normalize – a perfect match gives ~1.0 * 100
    max_possible = (len(req_skills_lower) * 3.0 + len(pref_skills_lower) * 1.5)
    if max_possible <= 0:
        max_possible = 10.0

    raw = total_weight / max_possible
    score = min(raw * 100, 100.0)

    # Penalty if ALL career history is in consulting firms
    all_consulting = all(
        any(c in entry.company.lower() for c in CONSULTING_COMPANIES)
        for entry in candidate.career_history
        if entry.company
    )
    if all_consulting:
        score *= 0.60  # 40% penalty

    return round(score, 2)


# ─────────────────────────────────────────────
# Behavioral signal scoring
# ─────────────────────────────────────────────

def _days_since(date_str: str) -> Optional[int]:
    """Return days since a date string, or None if unparseable."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (date.today() - dt).days
    except Exception:
        return None


def signal_score(candidate: Candidate) -> float:
    """
    Behavioral engagement score [0, 100] from redrob_signals.

    Key drivers (per hackathon JD):
      - recruiter_response_rate  (most important – availability signal)
      - last_active_date         (recency)
      - open_to_work_flag
      - interview_completion_rate
      - profile_completeness_score
      - github_activity_score    (technical credibility)
      - skill_assessment_scores  (verified competence)
      - notice_period_days       (availability speed)
    """
    s = candidate.redrob_signals
    parts: Dict[str, float] = {}

    # 1. Recruiter response rate (0–1 → 0–30 pts)
    parts["response_rate"] = s.recruiter_response_rate * 30.0

    # 2. Recency – last active (up to 20 pts; penalise >90d heavily)
    days_active = _days_since(s.last_active_date)
    if days_active is not None:
        if days_active <= 7:
            parts["recency"] = 20.0
        elif days_active <= 30:
            parts["recency"] = 18.0
        elif days_active <= 60:
            parts["recency"] = 12.0
        elif days_active <= 90:
            parts["recency"] = 7.0
        elif days_active <= 180:
            parts["recency"] = 3.0
        else:
            parts["recency"] = 0.0
    else:
        parts["recency"] = 5.0

    # 3. Open to work (5 pts)
    parts["open_to_work"] = 5.0 if s.open_to_work_flag else 0.0

    # 4. Interview completion rate (0–1 → 0–15 pts)
    parts["interview"] = s.interview_completion_rate * 15.0

    # 5. Profile completeness (0–100 → 0–10 pts)
    parts["profile_complete"] = s.profile_completeness_score / 10.0

    # 6. GitHub activity (-1 means not linked → 0; else 0-100 → 0-10 pts)
    gh = s.github_activity_score
    parts["github"] = (gh / 10.0) if gh >= 0 else 0.0

    # 7. Skill assessment scores – average of completed ones (0–5 pts)
    if s.skill_assessment_scores:
        avg_assess = sum(s.skill_assessment_scores.values()) / len(s.skill_assessment_scores)
        parts["assessments"] = (avg_assess / 100.0) * 5.0
    else:
        parts["assessments"] = 0.0

    # 8. Notice period bonus – sub-30 days preferred (up to 5 pts)
    if s.notice_period_days <= 15:
        parts["notice"] = 5.0
    elif s.notice_period_days <= 30:
        parts["notice"] = 4.0
    elif s.notice_period_days <= 60:
        parts["notice"] = 2.0
    else:
        parts["notice"] = 0.0

    raw = sum(parts.values())
    return round(min(raw, 100.0), 2)


# ─────────────────────────────────────────────
# Experience scoring
# ─────────────────────────────────────────────

def experience_score(candidate: Candidate, job_req: JobRequirements) -> float:
    """
    Score [0, 100] based on:
    - Years of experience vs JD range
    - Career progression relevance
    - Industry fit
    - Education tier
    - Avoidance of pure-consulting penalty
    """
    score = 0.0

    # 1. Years of experience (up to 35 pts)
    yoe = candidate.profile.years_of_experience
    # JD wants 5-9 years; 6-8 is ideal
    if 6 <= yoe <= 8:
        score += 35.0
    elif 5 <= yoe <= 9:
        score += 28.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        score += 18.0
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        score += 10.0
    else:
        score += 3.0

    # 2. Title relevance (up to 25 pts)
    title_lower = candidate.profile.current_title.lower()
    ai_ml_titles = {
        "ai engineer", "ml engineer", "machine learning engineer", "data scientist",
        "nlp engineer", "research engineer", "applied scientist", "applied ml",
        "senior engineer", "software engineer",
    }
    if any(t in title_lower for t in ai_ml_titles):
        score += 25.0
    elif "engineer" in title_lower or "developer" in title_lower:
        score += 15.0
    elif "data" in title_lower or "analyst" in title_lower:
        score += 8.0
    else:
        score += 0.0

    # 3. Career history quality (up to 25 pts)
    product_months = 0
    consulting_months = 0
    ai_role_months = 0
    for entry in candidate.career_history:
        co_lower = entry.company.lower()
        role_lower = entry.title.lower()
        is_consulting = any(c in co_lower for c in CONSULTING_COMPANIES)
        if is_consulting:
            consulting_months += entry.duration_months
        else:
            product_months += entry.duration_months

        if any(t in role_lower for t in ai_ml_titles):
            ai_role_months += entry.duration_months

    total_months = product_months + consulting_months
    if total_months > 0:
        product_ratio = product_months / total_months
        # More product company time = better
        score += product_ratio * 15.0

    # AI/ML role months bonus
    if ai_role_months >= 36:
        score += 10.0
    elif ai_role_months >= 18:
        score += 6.0
    elif ai_role_months >= 6:
        score += 3.0

    # 4. Education tier (up to 15 pts)
    edu_pts = 0.0
    for edu in candidate.education:
        tier = edu.tier
        if tier == "tier_1":
            edu_pts = max(edu_pts, 15.0)
        elif tier == "tier_2":
            edu_pts = max(edu_pts, 10.0)
        elif tier == "tier_3":
            edu_pts = max(edu_pts, 6.0)
        else:
            edu_pts = max(edu_pts, 2.0)
        # CS/Engg field bonus
        fos = edu.field_of_study.lower()
        if any(f in fos for f in ["computer", "software", "ai", "machine", "data"]):
            edu_pts = min(edu_pts + 3, 15.0)
    score += edu_pts

    return round(min(score, 100.0), 2)


# ─────────────────────────────────────────────
# Final composite scoring
# ─────────────────────────────────────────────

def final_score(
    semantic_score: float,   # 0-1
    sk_score: float,          # 0-100
    sig_score: float,         # 0-100
    exp_score: float,         # 0-100
) -> float:
    """
    Weighted combination per project spec:
      0.50 * semantic (normalised to 0-100)
      0.25 * skill_score
      0.15 * signal_score
      0.10 * experience_score
    Returns a score in [0, 1] (for CSV output compatibility).
    """
    sem_100 = semantic_score * 100.0
    raw = (
        0.50 * sem_100
        + 0.25 * sk_score
        + 0.15 * sig_score
        + 0.10 * exp_score
    )
    return round(raw / 100.0, 6)
