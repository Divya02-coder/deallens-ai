"""Evaluate supported regex clause categories against official CUAD JSON annotations.

CUAD is not bundled. Download the official dataset separately, then pass the
CUAD JSON file to this script. Only question categories mapped below are scored;
unmapped CUAD questions are intentionally excluded rather than mislabelled.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from contract_engine.extractor import regex_clause_scan, CLAUSE_PATTERNS

ALIASES={
 "change_of_control":("change of control","change in control"),
 "termination_convenience":("termination for convenience","termination without cause","terminate for convenience"),
 "exclusivity":("exclusivity","exclusive dealing","non-compete"),
 "auto_renewal":("automatic renewal","auto-renewal","automatically renew"),
 "uncapped_liability":("uncapped liability","unlimited liability"),
 "penalty":("liquidated damages","penalty"),
}
def map_category(question):
    q=re.sub(r"[^a-z0-9 -]"," ",question.lower())
    for category,aliases in ALIASES.items():
        if any(alias in q for alias in aliases): return category
    return None

def load_cases(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    cases=[]
    for article in payload.get("data",[]):
        title=article.get("title","unknown")
        for paragraph in article.get("paragraphs",[]):
            context=paragraph.get("context","")
            gold=set()
            for qa in paragraph.get("qas",[]):
                category=map_category(qa.get("question",""))
                if not category: continue
                answers=qa.get("answers",[])
                if answers and any(str(a.get("text","")).strip() for a in answers):
                    gold.add(category)
                elif qa.get("is_impossible") is False and answers:
                    gold.add(category)
            if gold or context:
                cases.append({"case_id":title,"text":context,"expected":gold})
    return cases

def evaluate(cases):
    categories=sorted(CLAUSE_PATTERNS)
    tp={c:0 for c in categories}; fp={c:0 for c in categories}; fn={c:0 for c in categories}
    citation_total=citation_supported=0
    for case in cases:
        hits=regex_clause_scan(case["text"])
        predicted={h["clause"] for h in hits}
        expected=case["expected"]
        for c in categories:
            tp[c]+=int(c in predicted and c in expected)
            fp[c]+=int(c in predicted and c not in expected)
            fn[c]+=int(c in expected and c not in predicted)
        # Regex returns a verbatim substring from the exact input context.
        for hit in hits:
            citation_total+=1
            citation_supported+=int(hit["quote"] and hit["quote"] in case["text"])
    rows=[]
    for c in categories:
        p=tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0
        r=tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0
        rows.append({"category":c,"tp":tp[c],"fp":fp[c],"fn":fn[c],
                     "precision":round(p,4),"recall":round(r,4),
                     "f1":round(2*p*r/(p+r),4) if p+r else 0})
    return {"cases":len(cases),"scored_categories":categories,"per_category":rows,
            "verbatim_quote_support":round(citation_supported/citation_total,4) if citation_total else None,
            "note":"Quote support means the regex excerpt is a verbatim substring; it does not prove the clause was interpreted correctly."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cuad_json",type=Path)
    ap.add_argument("--output",type=Path,default=Path("evaluation/cuad_results.json"))
    args=ap.parse_args()
    report=evaluate(load_cases(args.cuad_json))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
