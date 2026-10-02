from sentence_transformers import SentenceTransformer
from typing import List
from src.config import Config

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode(query, convert_to_numpy=True).tolist()

# Singleton
embedding_model = EmbeddingModel()