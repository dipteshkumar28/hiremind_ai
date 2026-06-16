"""
JD_ANALYZER node: reads the job description docx and uses Gemini
to extract structured requirements.
"""

from __future__ import annotations
import json
import os

from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from models.schemas import PipelineState, JobRequirements


JD_EXTRACTION_PROMPT = """
You are an expert technical recruiter analyst.

Read the following job description carefully and extract:

1. required_skills: A list of SPECIFIC technical skills that are MANDATORY for this role.
   Focus on: embeddings, vector databases, LLMs, fine-tuning, retrieval systems, Python, NLP, ranking systems.
   Keep each item short (2-5 words). Include at least 10 items.

2. preferred_skills: A list of NICE-TO-HAVE skills.
   Include: LoRA/QLoRA, learning-to-rank, HR tech experience, distributed systems, open-source.

3. experience_range: A string describing the target experience range (e.g. "5-9 years, ideally 6-8").

4. role_summary: A 2-3 sentence summary of what this role is about, emphasising the key technical mandate.

5. disqualifiers: List of profiles that should be explicitly penalised:
   - Pure consulting background (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini)
   - CV/speech/robotics focus without NLP
   - No production deployment experience
   - LLM experience < 12 months only
   - Title optimisers / frequent switchers

6. key_signals: Behavioral signals mentioned as important in the JD (e.g. "active on platform", "notice period < 30 days").

Return ONLY valid JSON with exactly these keys:
{{
  "required_skills": [...],
  "preferred_skills": [...],
  "experience_range": "...",
  "role_summary": "...",
  "disqualifiers": [...],
  "key_signals": [...]
}}

JOB DESCRIPTION:
{jd_text}
"""


def jd_analyzer_node(state: PipelineState) -> PipelineState:
    """LangGraph node: parse JD docx and extract structured requirements."""
    print("\n[JD_ANALYZER] Reading job description…")

    # Load JD from docx
    jd_path = os.path.join(os.path.dirname(__file__), "..", "data", "job_description.docx")
    doc = Document(jd_path)
    jd_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    state.jd_text = jd_text

    # Call Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1,
    )
    prompt = JD_EXTRACTION_PROMPT.replace(
    "{jd_text}",
    jd_text
    )    
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse JSON from response
    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)
    state.job_requirements = JobRequirements(**data)

    print(f"[JD_ANALYZER] Required skills: {state.job_requirements.required_skills}")
    print(f"[JD_ANALYZER] Experience range: {state.job_requirements.experience_range}")
    return state
