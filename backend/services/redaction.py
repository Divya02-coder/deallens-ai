"""Conservative PII redaction. Regex detection is dependency-free; NER is optional."""
from __future__ import annotations
import re, os
from typing import Any

PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)"),
    "pan_india": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I),
    "aadhaar_india": re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}(?!\d)"),
}
def redact_text(text: str, extra_patterns: dict[str, re.Pattern] | None = None) -> dict[str, Any]:
    """Return redacted text and type/count metadata; original values are never returned."""
    result=text
    counts={}
    patterns={**PATTERNS, **(extra_patterns or {})}
    for label, pattern in patterns.items():
        result, count=pattern.subn(f"[REDACTED_{label.upper()}]", result)
        if count: counts[label]=count
    # Optional NER: set DEALLENS_SPACY_MODEL to an installed spaCy model.
    # Only PERSON entities are redacted by default; organization names are often
    # material deal evidence and should not be removed automatically.
    model_name=os.getenv("DEALLENS_SPACY_MODEL")
    if model_name:
        try:
            import spacy
            nlp=spacy.load(model_name)
            doc=nlp(result)
            spans=[(ent.start_char,ent.end_char) for ent in doc.ents if ent.label_=="PERSON"]
            for start,end in sorted(spans,reverse=True):
                result=result[:start]+"[REDACTED_PERSON]"+result[end:]
            if spans: counts["person"]=counts.get("person",0)+len(spans)
        except (ImportError,OSError):
            # Regex redaction remains available when optional NER isn't installed.
            pass
    return {"text": result, "counts": counts, "redactions": sum(counts.values())}
