"""
FAISS vector store utilities for HireMind AI.
Uses Google Generative AI Embeddings with batched insertion.
"""

from __future__ import annotations
import os
import json
import pickle
from typing import List, Tuple, Dict

import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Singleton embedding model (re-used across calls)
_embed_model: GoogleGenerativeAIEmbeddings | None = None


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    global _embed_model
    if _embed_model is None:
        _embed_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    return _embed_model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return (N, D) float32 embeddings for a list of texts."""
    model = get_embedding_model()
    vectors = model.embed_documents(texts)
    return np.array(vectors, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """Return (D,) float32 embedding for a single query."""
    model = get_embedding_model()
    vec = model.embed_query(text)
    return np.array(vec, dtype="float32")


def create_vector_store(
    texts: List[str],
    ids: List[str],
    index_path: str,
    meta_path: str,
    batch_size: int = 100,
) -> faiss.IndexFlatIP:
    """
    Embed `texts` in batches and build a FAISS inner-product index
    (inner product == cosine similarity when vectors are L2-normalised).

    Saves index + id mapping to disk so they can be reloaded.
    Returns the in-memory index.
    """
    all_vecs: List[np.ndarray] = []
    total = len(texts)
    print(f"  Embedding {total} texts in batches of {batch_size}…")
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed_texts(batch)
        # L2-normalise for cosine similarity via inner product
        faiss.normalize_L2(vecs)
        all_vecs.append(vecs)
        if (i // batch_size) % 10 == 0:
            print(f"    {min(i + batch_size, total)}/{total}")

    matrix = np.vstack(all_vecs)  # (N, D)
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(ids, f)

    print(f"  FAISS index written to {index_path} ({index.ntotal} vectors, dim={dim})")
    return index


def load_vector_store(
    index_path: str, meta_path: str
) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """Load a previously saved FAISS index and id list."""
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        ids = pickle.load(f)
    return index, ids


def search_similar_candidates(
    query_text: str,
    index: faiss.IndexFlatIP,
    ids: List[str],
    top_k: int = 2000,
) -> List[Tuple[str, float]]:
    """
    Embed `query_text`, search the FAISS index, return
    [(candidate_id, similarity_score), …] sorted descending.
    """
    qvec = embed_query(query_text).reshape(1, -1)
    faiss.normalize_L2(qvec)

    scores, indices = index.search(qvec, min(top_k, index.ntotal))
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            results.append((ids[idx], float(score)))
    return results
