"""STEP 9 - PDF parsing with page numbers (every chunk keeps its source + page)."""
from pathlib import Path
import pymupdf


def parse_pdf(path: str) -> list[dict]:
    """Returns [{'source','page','text'}]. Empty text on every page => scanned PDF, use OCR."""
    doc = pymupdf.open(path)
    pages = [{"source": Path(path).name, "page": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]
    if not any(p["text"].strip() for p in pages):
        raise ValueError(f"{path} has no text layer (scanned?). Run OCR (pytesseract / cloud OCR) first.")
    return pages


def extract_tables(path: str, page_numbers: list[int] | None = None) -> list[dict]:
    import pdfplumber
    out = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if page_numbers and i not in page_numbers:
                continue
            for t in page.extract_tables():
                out.append({"source": Path(path).name, "page": i, "rows": t})
    return out


def chunk_pages(pages: list[dict], size: int = 900, overlap: int = 150) -> list[dict]:
    chunks = []
    for p in pages:
        t = p["text"]
        for start in range(0, max(len(t), 1), size - overlap):
            piece = t[start:start + size].strip()
            if piece:
                chunks.append({"source": p["source"], "page": p["page"], "text": piece})
    return chunks
