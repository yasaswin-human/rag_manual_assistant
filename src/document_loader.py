import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
from src.config import Config

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text and metadata from PDF.
    Returns list of pages with text and page number.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        # Basic section detection (can be improved)
        sections = detect_sections(text)
        pages.append({
            "page_num": page_num + 1,
            "text": text,
            "sections": sections,
        })
    doc.close()
    return pages

def detect_sections(text: str) -> List[str]:
    """
    Very naive section detection – improve with regex.
    """
    # Look for lines that look like headings (all caps, ending with ':', etc.)
    lines = text.split('\n')
    sections = []
    for line in lines:
        line = line.strip()
        if re.match(r'^[A-Z][A-Z\s]+$', line) or re.match(r'^[A-Z][a-z]+:.*$', line):
            sections.append(line)
    return sections

def chunk_document(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with metadata.
    """
    chunks = []
    for page in pages:
        text = page["text"]
        page_num = page["page_num"]
        # simple paragraph splitting
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # further split if too long?
            if len(para) > Config.CHUNK_SIZE:
                # split by sentences or fixed length
                sub_chunks = split_by_length(para, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
                for sub in sub_chunks:
                    chunks.append({
                        "text": sub,
                        "metadata": {
                            "page": page_num,
                            "source": "manual.pdf",  # will be filled later
                            "section": page.get("sections", [""])[0] if page.get("sections") else "",
                        }
                    })
            else:
                chunks.append({
                    "text": para,
                    "metadata": {
                        "page": page_num,
                        "source": "manual.pdf",
                        "section": page.get("sections", [""])[0] if page.get("sections") else "",
                    }
                })
    return chunks

def split_by_length(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks of approx chunk_size characters with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks