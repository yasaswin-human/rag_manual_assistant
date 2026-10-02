import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from src.embeddings import embedding_model

class VectorStore:
    def __init__(self, dimension: int = 384):  # BGE-small dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []  # list of dicts per vector

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Add documents (texts and metadata) to the index."""
        if len(texts) != len(metadatas):
            raise ValueError("Texts and metadatas must have same length")
        embeddings = embedding_model.embed_documents(texts)
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        self.metadata.extend(
            {"text": text, "metadata": metadata}
            for text, metadata in zip(texts, metadatas)
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k similar documents."""
        query_vec = np.array(embedding_model.embed_query(query)).astype('float32')
        distances, indices = self.index.search(query_vec.reshape(1, -1), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                results.append({
                    "text": self.metadata[idx].get("text", ""),
                    "metadata": self.metadata[idx].get("metadata", {}),
                    "score": distances[0][i],
                })
        return results

    def save(self, path: str):
        """Save index and metadata."""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        """Load index and metadata."""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def size(self) -> int:
        return self.index.ntotal