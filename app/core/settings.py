import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:8001")
PADDLE_EXTRACT_PATH_URL = os.environ.get(
    "PADDLE_EXTRACT_PATH_URL",
    "http://localhost:8003/paddle/pp-structurev3/predict_path",
)

DEFAULT_MODEL_NAME = "llama3.1:8b"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000

RAG_CHROMA_PERSIST_DIR = os.environ.get("RAG_CHROMA_PERSIST_DIR", "./chroma_db")
RAG_CHROMA_COLLECTION = os.environ.get("RAG_CHROMA_COLLECTION", "rag_default")
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "nomic-embed-text")
RAG_EMBEDDING_BASE_URL = os.environ.get("RAG_EMBEDDING_BASE_URL", OLLAMA_BASE_URL)
RAG_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "120"))
RAG_RETRIEVAL_TOP_K = int(os.environ.get("RAG_RETRIEVAL_TOP_K", "4"))
RAG_DEFAULT_KNOWLEDGE_DOMAIN = os.environ.get("RAG_DEFAULT_KNOWLEDGE_DOMAIN", "agriculture")
