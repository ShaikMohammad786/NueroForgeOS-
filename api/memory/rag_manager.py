# api/memory/rag_manager.py
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from .db_init import init_pinecone_client, init_embedding_model

# Pinecone index and embedder initialization
_pinecone_index = None
_embed_model = None

def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        _pinecone_index = init_pinecone_client()
    return _pinecone_index

def _get_embedder():
    global _embed_model
    if _embed_model is None:
        _embed_model = init_embedding_model()
    return _embed_model

def _embed(texts: List[str]):
    model = _get_embedder()
    return model.encode(texts, show_progress_bar=False).tolist()

# -------------------------------
# 🧩 Generic Vector DB Utilities
# -------------------------------

def _upsert_record(collection: str, text: str, metadata: Dict):
    """Upsert a single record into Pinecone under a namespace (collection)."""
    index = _get_index()
    rid = str(uuid.uuid4())
    emb = _embed([text])[0]

    # ✅ Clean metadata — remove None values and convert non-string-safe types
    clean_metadata = {}
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean_metadata[k] = v
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            clean_metadata[k] = v
        else:
            clean_metadata[k] = str(v)

    index.upsert(
        vectors=[
            {
                "id": rid,
                "values": emb,
                "metadata": clean_metadata
            }
        ],
        namespace=collection
    )
    return rid


def _query_records(collection: str, query: str, top_k: int = 4):
    """Query Pinecone namespace (collection) for semantic similarity."""
    index = _get_index()
    q_emb = _embed([query])[0]
    results = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True,
        namespace=collection
    )

    matches = []
    for match in results.get("matches", []):
        matches.append({
            "id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {})
        })
    return matches

# -------------------------------
# 🧰 Tools Collection
# -------------------------------

def add_tool(name: Optional[str], language: str, code: str, metadata: Optional[Dict] = None):
    created_at = datetime.utcnow().isoformat()
    metadata = metadata or {}
    metadata.update({"language": language, "name": name, "created_at": created_at})
    text_for_embed = (name or "") + "\n" + code[:8192]
    rid = _upsert_record("tools", text_for_embed, metadata)
    return rid

def retrieve_tools(query: str, top_k: int = 4):
    matches = _query_records("tools", query, top_k * 2)
    # Re-rank locally: prefer higher vector score, recent items, and success_count
    def score(m):
        md = m.get("metadata", {}) or {}
        vec = float(m.get("score", 0.0) or 0.0)
        success = float(md.get("success_count", 1) or 1)
        # Recent bonus
        created = md.get("created_at", "")
        recent_bonus = 0.0
        if created:
            # simple heuristic: newer gets small bonus
            recent_bonus = 0.05
        return vec + 0.2 * success + recent_bonus

    ranked = sorted(matches, key=score, reverse=True)
    return ranked[:top_k]

# -------------------------------
# ❌ Errors Collection
# -------------------------------

def add_error(error_text: str, stderr: Optional[str] = None, context: Optional[str] = None):
    created_at = datetime.utcnow().isoformat()
    metadata = {"stderr": stderr, "context": context, "created_at": created_at}
    text_for_embed = (error_text or "") + "\n" + (context or "")
    rid = _upsert_record("errors", text_for_embed, metadata)
    return rid

def retrieve_similar_errors(query: str, top_k: int = 4):
    return _query_records("errors", query, top_k)

# -------------------------------
# 📘 Docs Collection
# -------------------------------

def add_doc(title: str, content: str):
    created_at = datetime.utcnow().isoformat()
    metadata = {"title": title, "created_at": created_at}
    text_for_embed = title + "\n" + content[:8192]
    rid = _upsert_record("docs", text_for_embed, metadata)
    return rid

def retrieve_docs(query: str, top_k: int = 4):
    return _query_records("docs", query, top_k)

# -------------------------------
# 🧠 Patterns Collection
# -------------------------------

def add_pattern(name: str, content: str):
    created_at = datetime.utcnow().isoformat()
    metadata = {"name": name, "created_at": created_at}
    rid = _upsert_record("patterns", content[:8192], metadata)
    return rid

def retrieve_patterns(query: str, top_k: int = 4):
    return _query_records("patterns", query, top_k)

# -------------------------------
# 🛠 Fixes Collection (error -> fixed code)
# -------------------------------

def add_fix(error_signature: str, language: str, fixed_code: str, metadata: Optional[Dict] = None):
    """
    Persist a mapping from a normalized error signature to a fixed code variant.
    """
    created_at = datetime.utcnow().isoformat()
    metadata = metadata or {}
    metadata.update({"language": language, "created_at": created_at, "error_signature": error_signature})
    # Use signature as text to embed; include small code slice for context
    text_for_embed = f"{error_signature}\n{fixed_code[:2048]}"
    rid = _upsert_record("fixes", text_for_embed, metadata)
    return rid

def retrieve_fixes(error_signature_or_text: str, top_k: int = 2):
    """
    Retrieve candidate fixes by semantic similarity using the error signature or raw error text.
    """
    return _query_records("fixes", error_signature_or_text, top_k)