# Industrial Manual RAG Assistant

## Overview

Industrial Manual RAG Assistant indexes PDF equipment manuals and answers questions using retrieved manual text and a locally served Ollama language model. It is intended to make relevant maintenance information easier to find while keeping the generated response grounded in the indexed context.

## Key Features

- PDF text extraction with page metadata and basic heading detection
- Overlapping text chunking for long paragraphs
- Sentence Transformer embeddings and FAISS semantic search
- BM25 keyword search combined with semantic results using weighted, min-max-normalized scores
- Local LLM generation through the Ollama HTTP API
- FastAPI endpoints for PDF upload, querying, and index status
- Streamlit interface for uploading manuals, asking questions, and viewing retrieved excerpts with source, page, and section metadata
- An insufficiency heuristic and a retrieval-score-based confidence label

The model is prompted to answer from retrieved context and include page/manual references. The application displays retrieved source metadata and excerpts, but does not validate citations in generated answer text.

## Architecture

```text
PDF Manuals
    |
    v
Document Loading (PyMuPDF, page metadata)
    |
    v
Text Extraction and Chunking
    |
    v
Sentence Transformer Embeddings ------> BM25 Token Index
    |                                          |
    v                                          v
FAISS Semantic Search ----------------> Weighted Hybrid Retrieval
                                               |
                                               v
                                      Context and Source Metadata
                                               |
                                               v
                                         Ollama LLM
                                               |
                                               v
                              Grounded Answer and Retrieved Sources
                                      /                 \
                                     v                   v
                             FastAPI Backend      Streamlit Interface
```

## Project Structure

```text
.
|-- app/
|   |-- main.py
|   `-- streamlit_app.py
|-- scripts/
|   `-- index_docs.py
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- document_loader.py
|   |-- embeddings.py
|   |-- llm.py
|   |-- rag_pipeline.py
|   |-- retriever.py
|   |-- utils.py
|   `-- vector_store.py
|-- .env.example
`-- requirements.txt
```

The `data/manuals/` and `data/index/` paths are local runtime data directories; they are not source-controlled project files.

## Technology Stack

- Python
- FastAPI and Uvicorn
- Streamlit
- PyMuPDF for PDF text extraction
- Sentence Transformers for embeddings
- FAISS for vector search
- `rank-bm25` for keyword retrieval
- Ollama for local language-model generation
- NumPy, Pydantic, Requests, and python-dotenv

## Requirements

Python 3.10 or newer is recommended. The pinned NumPy dependency requires Python 3.10 or newer. Ollama must be installed separately to generate answers.

## Installation

Run these commands from PowerShell, replacing `<repository-url>` with the GitHub URL:

```powershell
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, activate it from Command Prompt with `.venv\Scripts\activate.bat`, or run the environment's Python directly as `.venv\Scripts\python.exe`.

## Ollama Setup

Install Ollama separately and ensure its service is reachable at the configured `OLLAMA_BASE_URL` (default `http://localhost:11434`). The default configured model is `llama3.2`:

```powershell
ollama pull llama3.2
```

Set `OLLAMA_MODEL` in `.env` if you use a different model. Ollama must remain running while the application generates answers.

## Indexing Documents

Place PDF manuals in a folder. From the repository root, pass that folder to the existing indexing script; for example:

```powershell
python scripts/index_docs.py .\data\manuals
```

The script scans PDFs directly inside the supplied folder (not recursively) and writes the FAISS index and BM25 corpus data under `INDEX_DIR` (default `./data/index`). This indexing command is optional if you plan to upload PDFs through the application.

## Running the Application

Start the FastAPI backend from the repository root:

```powershell
uvicorn app.main:app --reload
```

In a second terminal with the same virtual environment active, start Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

The Streamlit app sends requests to `http://localhost:8000`. The backend exposes `POST /index` for PDF upload, `POST /query` for questions, and `GET /status` for index status. The backend loads an existing index on startup when the persisted FAISS index and corpus are present.

## Configuration

Copy `.env.example` to `.env` and change values as needed. Configuration is loaded in `src/config.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `INDEX_DIR` | `./data/index` | Directory for the persisted FAISS index and corpus |
| `DATA_DIR` | `./data/manuals` | Declared default data path; currently not used by the indexing script, which accepts a folder argument |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence Transformer model identifier |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2` | Model sent to Ollama |
| `TOP_K` | `5` | Number of hybrid results included in context |
| `HYBRID_WEIGHT` | `0.5` | Semantic share of the weighted semantic/BM25 score |
| `CHUNK_SIZE` | `512` | Maximum approximate chunk length in characters |
| `CHUNK_OVERLAP` | `50` | Character overlap used when splitting long paragraphs |

## RAG Pipeline

PDF pages are extracted with PyMuPDF and divided into paragraph-based chunks. Long paragraphs are split into character-length pieces with overlap. The chunks are embedded and stored in FAISS alongside their text and page/source/section metadata; the indexing script also builds a BM25 corpus. At query time, semantic and BM25 candidates are each min-max normalized and combined with `HYBRID_WEIGHT`. Retrieved text is placed in the Ollama prompt, and source metadata and excerpts are returned separately. The confidence label is a simple threshold over the average combined retrieval score; it is not a calibrated probability.

## Limitations

- Answer generation depends on a locally installed, running Ollama service and an available model.
- Answers depend on the relevance and quality of the retrieved chunks and the PDF's extractable text; scanned/image-only pages are not OCR-processed.
- Heading detection and chunking are intentionally simple and may not preserve complex manual layouts or table structure.
- The confidence label is a retrieval-score heuristic, not a measure of factual correctness.
- The application keeps its active index and pipeline in process memory; this is a local assistant, not a multi-user persistence service.

## Future Improvements

- Add OCR and more robust extraction for tables and scanned manuals
- Evaluate retrieval quality and tune chunking and hybrid weights on representative manuals
- Validate answer citations against retrieved page/source metadata
- Add persistent, multi-user index management and configurable backend URLs in the interface

## License

License information will be added in a later step.