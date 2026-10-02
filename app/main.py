from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
import pickle
import tempfile
from src.document_loader import extract_text_from_pdf, chunk_document
from src.vector_store import VectorStore
from src.retriever import HybridRetriever
from src.rag_pipeline import RAGPipeline
from src.config import Config
import uuid
import json

app = FastAPI(title="Industrial Manual Assistant")

# Global state (in production, use a proper database)
vector_store = None
retriever = None
pipeline = None
indexed_files = {}  # mapping of file_id -> filename, etc.

@app.on_event("startup")
def startup():
    global vector_store, retriever, pipeline
    # Load existing index if available
    index_path = os.path.join(Config.INDEX_DIR, "faiss_index")
    if os.path.exists(f"{index_path}.faiss") and os.path.exists(f"{index_path}.pkl"):
        vector_store = VectorStore()
        vector_store.load(index_path)
        # Need to rebuild BM25 from stored metadata
        # For simplicity, we'll rebuild on each upload. We'll store corpus texts and metadata in a file.
        corpus_path = os.path.join(Config.INDEX_DIR, "corpus.pkl")
        if os.path.exists(corpus_path):
            with open(corpus_path, "rb") as f:
                corpus_data = pickle.load(f)
            texts, metadatas = corpus_data["texts"], corpus_data["metadatas"]
            retriever = HybridRetriever(vector_store)
            retriever.index(texts, metadatas)
            pipeline = RAGPipeline(retriever)

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict]] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    is_insufficient: bool
    confidence: str

@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    """Upload and index a PDF manual."""
    global vector_store, retriever, pipeline

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as buffer:
        temp_path = buffer.name
        shutil.copyfileobj(file.file, buffer)

    # Parse
    pages = extract_text_from_pdf(temp_path)
    chunks = chunk_document(pages)
    # Add source filename to metadata
    for chunk in chunks:
        chunk["metadata"]["source"] = file.filename

    # Keep the persisted BM25 corpus aligned with the cumulative FAISS index.
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    corpus_path = os.path.join(Config.INDEX_DIR, "corpus.pkl")
    if vector_store is not None and os.path.exists(corpus_path):
        with open(corpus_path, "rb") as f:
            corpus_data = pickle.load(f)
        texts = corpus_data["texts"] + texts
        metadatas = corpus_data["metadatas"] + metadatas

    # Initialize vector store if first time
    if vector_store is None:
        vector_store = VectorStore()
    vector_store.add_documents(texts, metadatas)

    # Build retriever
    retriever = HybridRetriever(vector_store)
    retriever.index(texts, metadatas)
    pipeline = RAGPipeline(retriever)

    # Save index and corpus
    os.makedirs(Config.INDEX_DIR, exist_ok=True)
    index_path = os.path.join(Config.INDEX_DIR, "faiss_index")
    vector_store.save(index_path)
    with open(corpus_path, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    # Clean temp
    os.remove(temp_path)

    return {"message": f"Indexed {file.filename} with {len(chunks)} chunks."}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a manual first.")
    result = pipeline.answer(request.query, request.chat_history)
    return result

@app.get("/status")
async def status():
    if vector_store is None:
        return {"indexed": False, "num_docs": 0}
    return {"indexed": True, "num_docs": vector_store.size()}