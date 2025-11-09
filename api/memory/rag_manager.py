import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from .db_init import init_chroma_client, init_embedding_model
from .schema import ToolRecord, ErrorRecord, DocRecord, PatternRecord

# collection names
COL_TOOLS = "tools"
COL_ERRORS = "errors"
COL_DOCS = "docs"
COL_PATTERNS = "patterns"

# init (singleton-ish)
_chroma_client = None
_embed_model = None

def _get_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = init_chroma_client()
    return _chroma_client

def _get_embedder():
    global _embed_model
    if _embed_model is None:
        _embed_model = init_embedding_model()
    return _embed_model

def _get_or_create_collection(name: str, metadata={}):
    client = _get_client()
    if name in [c.name for c in client.list_collections()]:
        return client.get_collection(name)
    return client.create_collection(name=name, metadata=metadata)

def _embed(texts: List[str]):
    model = _get_embedder()
    return model.encode(texts, show_progress_bar=False).tolist()

# ----- Tools -----
def add_tool(name: Optional[str], language: str, code: str, metadata: Optional[Dict]=None):
    col = _get_or_create_collection(COL_TOOLS)
    rid = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    metadata = metadata or {}
    metadata.update({"language": language, "name": name, "created_at": created_at})
    text_for_embed = (name or "") + "\n" + code[:8192]  # limit length
    emb = _embed([text_for_embed])[0]
    col.add(ids=[rid], documents=[code], metadatas=[metadata], embeddings=[emb])
    return rid

def retrieve_tools(query: str, top_k: int = 4):
    col = _get_or_create_collection(COL_TOOLS)
    emb = _embed([query])[0]
    results = col.query(query_embeddings=[emb], n_results=top_k)
    # results: dict with ids, documents, metadatas
    out = []
    for i, doc in enumerate(results.get("documents", [[]])[0]):
        out.append({
            "id": results["ids"][0][i],
            "code": doc,
            "metadata": results["metadatas"][0][i],
            "score": results.get("distances", [[None]*top_k])[0][i]
        })
    return out

# ----- Errors -----
def add_error(error_text: str, stderr: Optional[str]=None, context: Optional[str]=None):
    col = _get_or_create_collection(COL_ERRORS)
    rid = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    metadata = {"created_at": created_at}
    text_for_embed = (error_text or "") + "\n" + (context or "")
    emb = _embed([text_for_embed])[0]
    col.add(ids=[rid], documents=[error_text], metadatas=[metadata], embeddings=[emb])
    return rid

def retrieve_similar_errors(query: str, top_k: int = 4):
    col = _get_or_create_collection(COL_ERRORS)
    emb = _embed([query])[0]
    results = col.query(query_embeddings=[emb], n_results=top_k)
    out = []
    for i, doc in enumerate(results.get("documents", [[]])[0]):
        out.append({"id": results["ids"][0][i], "error_text": doc, "metadata": results["metadatas"][0][i]})
    return out

# ----- Docs & Patterns - similar utilities -----
def add_doc(title: str, content: str):
    col = _get_or_create_collection(COL_DOCS)
    rid = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    text_for_embed = title + "\n" + content[:8192]
    emb = _embed([text_for_embed])[0]
    metadata = {"title": title, "created_at": created_at}
    col.add(ids=[rid], documents=[content], metadatas=[metadata], embeddings=[emb])
    return rid

def retrieve_docs(query: str, top_k: int = 4):
    col = _get_or_create_collection(COL_DOCS)
    emb = _embed([query])[0]
    results = col.query(query_embeddings=[emb], n_results=top_k)
    out = []
    for i, doc in enumerate(results.get("documents", [[]])[0]):
        out.append({"id": results["ids"][0][i], "title": results["metadatas"][0][i].get("title"), "content": doc})
    return out
