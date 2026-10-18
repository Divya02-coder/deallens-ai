"""Evaluate category-level precision/recall/F1 from labelled JSONL cases."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def score(records):
    categories=sorted({str(c) for r in records for c in (r.get("expected_categories",[])+r.get("predicted_categories",[]))})
    rows=[]
    for category in categories:
        tp=fp=fn=0
        for r in records:
            expected=set(r.get("expected_categories",[])); predicted=set(r.get("predicted_categories",[]))
            tp += category in expected and category in predicted
            fp += category not in expected and category in predicted
            fn += category in expected and category not in predicted
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        rows.append({"category":category,"tp":tp,"fp":fp,"fn":fn,
                     "precision":round(precision,4),"recall":round(recall,4),"f1":round(f1,4)})
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("labels",type=Path,help="JSONL: case_id, expected_categories, predicted_categories")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    records=[json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    results=score(records)
    payload={"cases":len(records),"per_category":results}
    text=json.dumps(payload,indent=2)
    if args.output: args.output.write_text(text+"\n",encoding="utf-8")
    print(text)
if __name__=="__main__": main()
