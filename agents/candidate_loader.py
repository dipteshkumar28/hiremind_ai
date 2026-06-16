"""
CANDIDATE_LOADER node: streams candidates.jsonl, validates schema, 
and populates state.candidates.
"""

from __future__ import annotations
import json
import os
from typing import List

from tqdm import tqdm
from pydantic import ValidationError

from models.schemas import PipelineState, Candidate


def candidate_loader_node(state: PipelineState) -> PipelineState:
    """LangGraph node: load and validate all 100K candidate records."""
    jsonl_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "candidates.jsonl"
    )
    print(f"\n[CANDIDATE_LOADER] Loading candidates from {jsonl_path}…")

    candidates: List[Candidate] = []
    errors = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="  Loading", unit=" candidates"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                candidate = Candidate.model_validate(raw)
                candidates.append(candidate)
            except (json.JSONDecodeError, ValidationError) as e:
                errors += 1
                if errors <= 5:
                    print(f"  [WARN] Line {line_num}: {e}")

    print(f"[CANDIDATE_LOADER] Loaded {len(candidates):,} candidates ({errors} errors skipped)")
    state.candidates = candidates
    return state
