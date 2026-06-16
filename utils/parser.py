"""
Utilities for parsing candidate data into rich text for embedding.
"""

from __future__ import annotations
from models.schemas import Candidate


def build_candidate_profile_text(candidate: Candidate) -> str:
    """
    Create a rich, structured text representation of a candidate
    suitable for semantic embedding.

    Format mirrors what EMBEDDING_AGENT specifies:
      Title | Summary | Skills | Experience | Career History
    """
    p = candidate.profile
    lines = []

    # --- Title / Headline ---
    lines.append(f"TITLE: {p.current_title}")
    if p.headline:
        lines.append(f"HEADLINE: {p.headline}")

    # --- Summary ---
    if p.summary:
        lines.append(f"\nSUMMARY:\n{p.summary}")

    # --- Skills ---
    if candidate.skills:
        skill_parts = []
        for sk in candidate.skills:
            skill_parts.append(f"{sk.name} ({sk.proficiency}, {sk.duration_months}mo, {sk.endorsements} endorsements)")
        lines.append(f"\nSKILLS:\n" + "; ".join(skill_parts))

    # --- Skill assessments ---
    assessments = candidate.redrob_signals.skill_assessment_scores
    if assessments:
        assess_str = ", ".join(f"{k}: {v:.0f}/100" for k, v in assessments.items())
        lines.append(f"VERIFIED ASSESSMENTS: {assess_str}")

    # --- Experience ---
    lines.append(f"\nEXPERIENCE: {p.years_of_experience} years")
    lines.append(f"CURRENT: {p.current_title} at {p.current_company} ({p.current_industry})")
    lines.append(f"LOCATION: {p.location}, {p.country}")

    # --- Career History ---
    if candidate.career_history:
        lines.append("\nCAREER HISTORY:")
        for entry in candidate.career_history:
            end = entry.end_date if entry.end_date else "Present"
            lines.append(
                f"  [{entry.start_date[:7]} – {end[:7] if end != 'Present' else end}] "
                f"{entry.title} @ {entry.company} ({entry.industry}, {entry.company_size}) "
                f"[{entry.duration_months}mo]"
            )
            if entry.description:
                # Include first 300 chars of description
                desc = entry.description[:300].replace("\n", " ")
                lines.append(f"    {desc}")

    # --- Education ---
    if candidate.education:
        edu_parts = []
        for edu in candidate.education:
            edu_parts.append(
                f"{edu.degree} in {edu.field_of_study} from {edu.institution} ({edu.tier})"
            )
        lines.append(f"\nEDUCATION: {'; '.join(edu_parts)}")

    # --- Certifications ---
    if candidate.certifications:
        cert_parts = [f"{c.name} by {c.issuer} ({c.year})" for c in candidate.certifications]
        lines.append(f"CERTIFICATIONS: {'; '.join(cert_parts)}")

    # --- GitHub ---
    gh = candidate.redrob_signals.github_activity_score
    if gh >= 0:
        lines.append(f"GITHUB ACTIVITY SCORE: {gh:.1f}/100")

    return "\n".join(lines)
