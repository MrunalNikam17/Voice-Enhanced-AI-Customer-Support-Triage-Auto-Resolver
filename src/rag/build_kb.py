"""
src/rag/build_kb.py

Builds the RAG knowledge base for the Responder Agent:
1. Loads curated FAQ entries (data/raw/knowledge_base/knowledge_base_faq.json)
2. Embeds them with a sentence-transformer
3. Stores them in a FAISS vector index (models/kb_store/)

Run from project root:
    python -m src.rag.build_kb
"""

import json
import os
import pickle
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())
from src.utils.config import config


def load_faq(path: str = None) -> list:
    path = path or config.KB_JSON_PATH
    with open(path, "r") as f:
        return json.load(f)


def build_vector_store(faq_entries: list, out_dir: str = None):
    out_dir = out_dir or config.KB_STORE_DIR
    os.makedirs(out_dir, exist_ok=True)

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    texts = [f"{e['title']}. {e['content']}" for e in faq_entries]
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(out_dir, "faq.index"))
    with open(os.path.join(out_dir, "faq_metadata.pkl"), "wb") as f:
        pickle.dump(faq_entries, f)

    print(f"Indexed {len(faq_entries)} FAQ entries -> {out_dir}/")
    return index, embedder, faq_entries


def load_vector_store(store_dir: str = None):
    """Loads a previously built index + metadata + embedder for retrieval."""
    store_dir = store_dir or config.KB_STORE_DIR
    index = faiss.read_index(os.path.join(store_dir, "faq.index"))
    with open(os.path.join(store_dir, "faq_metadata.pkl"), "rb") as f:
        faq_entries = pickle.load(f)
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return index, embedder, faq_entries


def retrieve(query: str, index, embedder, faq_entries: list,
             top_k: int = None, min_score: float = None) -> list:
    """Used by the Responder Agent (src/responder/responder.py)."""
    top_k = top_k or config.RAG_TOP_K
    min_score = min_score if min_score is not None else config.RAG_MIN_SCORE

    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")

    scores, idxs = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1 or score < min_score:
            continue
        entry = faq_entries[idx]
        results.append({
            "id": entry["id"],
            "category": entry["category"],
            "title": entry["title"],
            "content": entry["content"],
            "score": float(score),
        })
    return results


if __name__ == "__main__":
    faq_entries = load_faq()
    index, embedder, faq_entries = build_vector_store(faq_entries)

    test_query = "I was charged twice this month, what should I do?"
    results = retrieve(test_query, index, embedder, faq_entries)

    print(f"\nQuery: {test_query}\n")
    for r in results:
        print(f"[{r['score']:.3f}] ({r['category']}) {r['title']}")
        print(f"    -> {r['content']}\n")
