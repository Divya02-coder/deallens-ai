"""Attach traceable document/table snippets to risk findings without inventing support."""
from __future__ import annotations
import re

def _terms(finding):
    text=" ".join(str(finding.get(k,"")) for k in ("title","category","detail"))
    terms={x.lower() for x in re.findall(r"[a-zA-Z]{4,}",text)}
    return terms - {"with","from","that","this","within","exposure","potential","risk","analysis","findings"}

def fuse_evidence(findings: list[dict], chunks: list[dict], max_per_finding: int=2) -> list[dict]:
    out=[]
    for original in findings:
        f=dict(original)
        terms=_terms(f)
        candidates=[]
        for ch in chunks:
            body=str(ch.get("text",""))
            low=body.lower()
            score=sum(1 for term in terms if term in low)
            if score:
                candidates.append((score,ch))
        candidates.sort(key=lambda item:item[0],reverse=True)
        fused=[]
        for score,ch in candidates[:max_per_finding]:
            fused.append({
                "source": ch.get("source","unknown"),
                "page": ch.get("page"),
                "location": ch.get("location"),
                "row_start": ch.get("row_start"),
                "row_end": ch.get("row_end"),
                "passage": str(ch.get("text",""))[:700],
                "match_terms": score,
                "verification_status": "retrieval_match_not_independent_verification",
            })
        f["document_evidence"]=fused
        out.append(f)
    return out
