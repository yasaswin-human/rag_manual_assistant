import math
from typing import List, Dict, Any
from src.vector_store import VectorStore
from rank_bm25 import BM25Okapi
from src.config import Config

class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25_index = None
        self.corpus_texts = None
        self.metadata_list = None

    def index(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Build BM25 index from the same corpus."""
        self.corpus_texts = texts
        self.metadata_list = metadatas
        tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        # simple tokenization (lowercase, split on whitespace)
        return text.lower().split()

    def retrieve(self, query: str, k: int = Config.TOP_K) -> List[Dict[str, Any]]:
        """Retrieve using hybrid search: combine scores from vector and BM25."""
        # Semantic search
        semantic_results = self.vector_store.search(query, k=5*k)  # get more to rerank

        # BM25 search
        if self.bm25_index is None:
            bm25_results = []
        else:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            # Get top k indices
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5*k]
            bm25_results = []
            for idx in top_indices:
                bm25_results.append({
                    "text": self.corpus_texts[idx],
                    "metadata": self.metadata_list[idx],
                    "score": bm25_scores[idx],
                })

        # Combine scores (normalize both to [0,1] and apply weight)
        combined = self._combine_results(semantic_results, bm25_results, weight=Config.HYBRID_WEIGHT)
        # Sort by combined score and return top k
        combined.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined[:k]

    def _combine_results(self, semantic: List[Dict], bm25: List[Dict], weight: float) -> List[Dict]:
        """Reciprocal rank fusion or weighted sum. Here we use weighted sum of normalized scores."""
        # Normalize scores within each list (min-max)
        semantic_scores = [r["score"] for r in semantic] if semantic else []
        bm25_scores = [r["score"] for r in bm25] if bm25 else []

        # For FAISS L2, smaller is better -> convert to similarity (1 - normalized distance)
        if semantic_scores:
            min_s, max_s = min(semantic_scores), max(semantic_scores)
            if max_s > min_s:
                norm_semantic = [1 - (s - min_s) / (max_s - min_s) for s in semantic_scores]
            else:
                norm_semantic = [1.0] * len(semantic_scores)
        else:
            norm_semantic = []

        if bm25_scores:
            min_b, max_b = min(bm25_scores), max(bm25_scores)
            if max_b > min_b:
                norm_bm25 = [(s - min_b) / (max_b - min_b) for s in bm25_scores]
            else:
                norm_bm25 = [1.0] * len(bm25_scores)
        else:
            norm_bm25 = []

        # Create combined list: merge by text or metadata? We'll use a dictionary keyed by text (or index)
        # For simplicity, we'll merge by position in the lists (assuming same order if they align)
        # But they may not align. Better: create a set of all texts.
        combined_dict = {}
        # Add semantic
        for i, res in enumerate(semantic):
            key = res["text"]  # assuming text is unique enough
            combined_dict[key] = {
                "text": res["text"],
                "metadata": res["metadata"],
                "semantic_score": norm_semantic[i] if i < len(norm_semantic) else 0.0,
                "bm25_score": 0.0,
            }
        # Add BM25
        for i, res in enumerate(bm25):
            key = res["text"]
            if key in combined_dict:
                combined_dict[key]["bm25_score"] = norm_bm25[i] if i < len(norm_bm25) else 0.0
            else:
                combined_dict[key] = {
                    "text": res["text"],
                    "metadata": res["metadata"],
                    "semantic_score": 0.0,
                    "bm25_score": norm_bm25[i] if i < len(norm_bm25) else 0.0,
                }

        # Compute combined score
        result = []
        for key, item in combined_dict.items():
            combined_score = weight * item["semantic_score"] + (1 - weight) * item["bm25_score"]
            result.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "combined_score": combined_score,
                "semantic_score": item["semantic_score"],
                "bm25_score": item["bm25_score"],
            })
        return result