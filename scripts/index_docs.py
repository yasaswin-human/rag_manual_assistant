import os
import sys
import glob
from src.document_loader import extract_text_from_pdf, chunk_document
from src.vector_store import VectorStore
from src.retriever import HybridRetriever
from src.config import Config
import pickle

def index_folder(folder_path: str):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        print("No PDFs found.")
        return

    vector_store = VectorStore()
    all_texts = []
    all_metadatas = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename}...")
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_document(pages)
        for chunk in chunks:
            chunk["metadata"]["source"] = filename
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        all_texts.extend(texts)
        all_metadatas.extend(metadatas)

    if not all_texts:
        print("No text extracted.")
        return

    vector_store.add_documents(all_texts, all_metadatas)
    retriever = HybridRetriever(vector_store)
    retriever.index(all_texts, all_metadatas)

    # Save
    os.makedirs(Config.INDEX_DIR, exist_ok=True)
    index_path = os.path.join(Config.INDEX_DIR, "faiss_index")
    vector_store.save(index_path)
    corpus_path = os.path.join(Config.INDEX_DIR, "corpus.pkl")
    with open(corpus_path, "wb") as f:
        pickle.dump({"texts": all_texts, "metadatas": all_metadatas}, f)

    print(f"Indexed {len(all_texts)} chunks from {len(pdf_files)} PDFs.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_docs.py <path_to_manuals_folder>")
        sys.exit(1)
    index_folder(sys.argv[1])