"""
Pydantic models adapted exactly to the Redrob candidate schema.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Sub-models for career_history
# ─────────────────────────────────────────────

class CareerEntry(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = "unknown"   # tier_1 … tier_4, unknown


class Skill(BaseModel):
    name: str
    proficiency: str = "beginner"   # beginner | intermediate | advanced | expert
    endorsements: int = 0
    duration_months: int = 0


class Certification(BaseModel):
    name: str
    issuer: str
    year: int


class Language(BaseModel):
    language: str
    proficiency: str = "basic"


class SalaryRange(BaseModel):
    min: float = 0.0
    max: float = 0.0


class RedrobSignals(BaseModel):
    profile_completeness_score: float = 0.0
    signup_date: str = ""
    last_active_date: str = ""
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: Dict[str, float] = Field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 90
    expected_salary_range_inr_lpa: SalaryRange = Field(default_factory=SalaryRange)
    preferred_work_mode: str = "flexible"
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


class Profile(BaseModel):
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""


# ─────────────────────────────────────────────
# Top-level Candidate model
# ─────────────────────────────────────────────

class Candidate(BaseModel):
    candidate_id: str
    profile: Profile
    career_history: List[CareerEntry] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)


# ─────────────────────────────────────────────
# JD extraction model
# ─────────────────────────────────────────────

class JobRequirements(BaseModel):
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience_range: str = ""
    role_summary: str = ""
    disqualifiers: List[str] = Field(default_factory=list)
    key_signals: List[str] = Field(default_factory=list)  # behavioral hints from JD


# ─────────────────────────────────────────────
# Scoring models
# ─────────────────────────────────────────────

class CandidateScore(BaseModel):
    candidate_id: str
    semantic_score: float = 0.0      # 0-1  (from FAISS cosine similarity)
    skill_score: float = 0.0         # 0-100
    signal_score: float = 0.0        # 0-100  (behavioral)
    experience_score: float = 0.0    # 0-100
    final_score: float = 0.0         # weighted composite


class RankingResult(BaseModel):
    candidate_id: str
    rank: int
    score: float                     # matches submission column "score"
    reasoning: str


# ─────────────────────────────────────────────
# LangGraph state
# ─────────────────────────────────────────────

class PipelineState(BaseModel):
    """Shared state passed between LangGraph nodes."""
    jd_text: str = ""
    job_requirements: Optional[JobRequirements] = None
    candidates: List[Candidate] = Field(default_factory=list)
    # Maps candidate_id -> FAISS index position (populated by embedding agent)
    id_to_index: Dict[str, int] = Field(default_factory=dict)
    # FAISS index object cannot be serialized – stored as a sidecar
    faiss_index_path: str = ""
    # Top-N candidate ids from semantic search
    top_candidate_ids: List[str] = Field(default_factory=list)
    # Scored candidates
    scores: List[CandidateScore] = Field(default_factory=list)
    # Final ranked results
    results: List[RankingResult] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
