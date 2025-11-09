import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download

print("✅ patched huggingface_hub.cached_download")


def init_chroma_client():
    chroma_url = "http://localhost:8000"
    chroma_key = "ck-2p3kqnVduTxppheFk2vSoAa9DENWTFbPHNEAKMxSvPhm"
    team_id = os.getenv("CHROMA_TEAM_ID", "ce37940f-7dd9-47bd-8870-f46b3ff846ed").strip()

    headers = {}
    if chroma_key:
        if chroma_key.startswith("ck-"):
            headers["Authorization"] = f"Bearer {chroma_key}"
            if team_id:
                headers["X-Chroma-Team-Id"] = team_id
        else:
            headers["X-API-Key"] = chroma_key

    print(f"🌐 Connecting to Chroma ({chroma_url})...")

    # ✅ Don't mix `Settings` and `host` arguments
    client = chromadb.HttpClient(
        host=chroma_url,
        headers=headers
    )

    print(f"✅ Connected to Chroma: {chroma_url}")
    return client

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

def init_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)