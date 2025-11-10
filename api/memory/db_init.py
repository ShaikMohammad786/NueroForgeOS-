# api/memory/db_init.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def init_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "neuroforge-memory")
    region = os.getenv("PINECONE_ENV", "us-east-1")

    if not api_key:
        raise ValueError("❌ Missing PINECONE_API_KEY in .env")

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    # Ensure index exists
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        print(f"⚙️ Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region)
        )

    index = pc.Index(index_name)
    print(f"✅ Connected to Pinecone index: {index_name}")
    return index

def init_embedding_model():
    model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    print(f"🧠 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model
