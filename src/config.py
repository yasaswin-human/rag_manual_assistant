import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    DATA_DIR = os.getenv("DATA_DIR", "./data/manuals")
    INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")

    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # LLM via Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # or mistral, qwen, etc.

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", 5))
    HYBRID_WEIGHT = float(os.getenv("HYBRID_WEIGHT", 0.5))  # between semantic and BM25

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))