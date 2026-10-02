from typing import List, Dict, Any, Optional
from src.retriever import HybridRetriever
from src.llm import llm
from src.config import Config
import re

class RAGPipeline:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def answer(self, query: str, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Full pipeline: retrieve, build context, generate answer, check sufficiency.
        Returns dict with answer, sources, confidence, etc.
        """
        # 1. Retrieve
        retrieved_docs = self.retriever.retrieve(query, k=Config.TOP_K)

        # 2. Build context with source metadata
        context_texts = []
        sources = []
        for doc in retrieved_docs:
            context_texts.append(doc["text"])
            sources.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": doc["combined_score"],
            })

        context = "\n\n---\n\n".join(context_texts)

        # 3. Create prompt with system instructions (safety-first)
        system_prompt = (
            "You are an assistant for industrial equipment maintenance. "
            "You must only answer based on the provided context. "
            "If the context does not contain enough information to answer the question, "
            "respond with: 'The manual does not provide enough information.' "
            "Do not invent any procedures, specifications, torque values, safety limits, or replacement parts. "
            "Always cite the source page and manual when using information."
        )

        user_prompt = f"""
Context:
{context}

Question: {query}

Based on the context above, provide a grounded answer. Include specific citations (page number and manual). If the context is insufficient, say so clearly.
"""

        # 4. Call LLM
        answer = llm.generate(user_prompt, system_prompt=system_prompt)

        # 5. Check if answer indicates insufficiency or if confidence is low
        # Simple heuristic: if answer contains "does not provide enough" or similar
        insufficient_indicators = ["does not provide enough", "not enough information", "manual does not contain", "insufficient"]
        is_insufficient = any(ind in answer.lower() for ind in insufficient_indicators)

        # Also compute a confidence score based on retrieval scores
        if retrieved_docs:
            avg_score = sum(d["combined_score"] for d in retrieved_docs) / len(retrieved_docs)
            confidence = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
        else:
            confidence = "No evidence"

        # 6. Extract citations from answer? We already have sources.
        # We'll return the answer and sources separately.

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "is_insufficient": is_insufficient,
            "confidence": confidence,
            "retrieved_docs": retrieved_docs,
        }