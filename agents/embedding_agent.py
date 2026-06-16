"""
EMBEDDING_AGENT node: converts candidate profiles to text, embeds them,
and stores in a FAISS index.

To avoid re-embedding 100K candidates on every run, the index is cached
to disk and reloaded if present.
"""

from __future__ import annotations
import os
from typing import List

from utils.parser import build_candidate_profile_text
from utils.faiss_store import create_vector_store, load_vector_store
from models.schemas import PipelineState

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "candidates.faiss")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "candidates_ids.pkl")

EMBED_BATCH_SIZE = 100   # Google Embeddings API batch limit


def embedding_agent_node(state: PipelineState) -> PipelineState:
    """LangGraph node: embed all candidates and store in FAISS."""

    # ── Cache check ──────────────────────────────────────────────────────────
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print(f"\n[EMBEDDING_AGENT] Loading cached FAISS index from {INDEX_PATH}…")
        index, ids = load_vector_store(INDEX_PATH, META_PATH)
        print(f"[EMBEDDING_AGENT] Loaded {index.ntotal:,} vectors (dim={index.d})")
        state.faiss_index_path = INDEX_PATH
        # Build id→index map for later lookup
        state.id_to_index = {cid: i for i, cid in enumerate(ids)}
        return state

    # ── Build profile texts ───────────────────────────────────────────────────
    print(f"\n[EMBEDDING_AGENT] Building profile texts for {len(state.candidates):,} candidates…")
    texts: List[str] = []
    ids: List[str] = []
    for c in state.candidates:
        texts.append(build_candidate_profile_text(c))
        ids.append(c.candidate_id)

    # ── Embed + FAISS ─────────────────────────────────────────────────────────
    print(f"[EMBEDDING_AGENT] Embedding {len(texts):,} texts in batches of {EMBED_BATCH_SIZE}…")
    create_vector_store(
        texts=texts,
        ids=ids,
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        batch_size=EMBED_BATCH_SIZE,
    )

    state.faiss_index_path = INDEX_PATH
    state.id_to_index = {cid: i for i, cid in enumerate(ids)}
    print(f"[EMBEDDING_AGENT] Done. FAISS index saved to {INDEX_PATH}")
    return state
