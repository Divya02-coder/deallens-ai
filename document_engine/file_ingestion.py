"""Multi-format evidence ingestion for DealLens AI.

Supported: PDF, CSV, XLSX/XLS, TXT, MD, JSON and DOCX.
Every generated chunk keeps source metadata so answers can cite evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from document_engine.pdf_parser import parse_pdf, chunk_pages

SUPPORTED = {".pdf", ".csv", ".xlsx", ".xls", ".txt", ".md", ".json", ".docx"}


def _table_chunks(source: str, df: pd.DataFrame, sheet: str | None = None, rows_per_chunk: int = 25) -> list[dict]:
    out = []
    frame = df.copy()
    frame.columns = [str(c) for c in frame.columns]
    for start in range(0, len(frame), rows_per_chunk):
        part = frame.iloc[start:start + rows_per_chunk]
        text = part.to_csv(index=False)
        if not text.strip():
            continue
        out.append({
            "source": source,
            "page": 1,
            "location": f"sheet={sheet}" if sheet else "table",
            "row_start": int(start) + 1,
            "row_end": int(start + len(part)),
            "text": text,
        })
    if not out and len(frame.columns):
        out.append({"source": source, "page": 1, "location": f"sheet={sheet}" if sheet else "table", "text": ", ".join(frame.columns)})
    return out


def ingest_file(path: str) -> tuple[list[dict], dict]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}. Supported: {', '.join(sorted(SUPPORTED))}")

    if ext == ".pdf":
        chunks = chunk_pages(parse_pdf(str(p)))
        return chunks, {"type": "pdf", "source": p.name, "records": len(chunks)}

    if ext in {".csv", ".xlsx", ".xls"}:
        chunks = []
        sheets = {}
        if ext == ".csv":
            df = pd.read_csv(p)
            sheets["csv"] = df
            chunks.extend(_table_chunks(p.name, df))
        else:
            sheets = pd.read_excel(p, sheet_name=None)
            for sheet, df in sheets.items():
                chunks.extend(_table_chunks(p.name, df, sheet=str(sheet)))
        return chunks, {"type": "spreadsheet", "source": p.name, "sheets": list(sheets), "records": sum(len(x) for x in sheets.values())}

    if ext in {".txt", ".md"}:
        text = p.read_text(encoding="utf-8", errors="replace")
        size, overlap = 1200, 150
        chunks = []
        for start in range(0, max(len(text), 1), size - overlap):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"source": p.name, "page": 1, "location": "text", "text": piece})
        return chunks, {"type": "text", "source": p.name, "records": len(chunks)}

    if ext == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        text = json.dumps(raw, indent=2, ensure_ascii=False, default=str)
        return [{"source": p.name, "page": 1, "location": "json", "text": text[:12000]}], {"type": "json", "source": p.name, "records": 1}

    if ext == ".docx":
        from docx import Document
        doc = Document(str(p))
        paragraphs = [x.text.strip() for x in doc.paragraphs if x.text.strip()]
        text = "\n".join(paragraphs)
        chunks = []
        size, overlap = 1200, 150
        for start in range(0, max(len(text), 1), size - overlap):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"source": p.name, "page": 1, "location": "docx", "text": piece})
        return chunks, {"type": "docx", "source": p.name, "records": len(chunks)}

    raise ValueError("Unsupported file type")
