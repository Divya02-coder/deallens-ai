"""Benchmark an Ollama model for latency/token throughput; run multiple models separately."""
from __future__ import annotations
import argparse, json, os, statistics, time
from pathlib import Path
from backend.services.model_provider import generate_local

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default=os.getenv("OLLAMA_MODEL","qwen2.5:7b"))
    parser.add_argument("--prompts",type=Path,required=True,help="JSON array of {id,prompt,system_instruction?,reference?}")
    parser.add_argument("--output",type=Path,default=Path("evaluation/ollama_results.json"))
    args=parser.parse_args()
    prompts=json.loads(args.prompts.read_text(encoding="utf-8"))
    results=[]
    for item in prompts:
        start=time.perf_counter()
        response=generate_local(item["prompt"],item.get("system_instruction",""),args.model)
        results.append({"id":item.get("id"),"model":response["model"],"latency_ms":response["latency_ms"],
                        "prompt_tokens":response.get("prompt_eval_count"),"completion_tokens":response.get("eval_count"),
                        "output":response["text"],"reference":item.get("reference")})
    latencies=[r["latency_ms"] for r in results]
    report={"model":args.model,"cases":len(results),"mean_latency_ms":round(statistics.mean(latencies),2) if latencies else None,
            "median_latency_ms":round(statistics.median(latencies),2) if latencies else None,
            "results":results,
            "notes":"Review outputs against references for quality. Memory use and cost are deployment-specific; record Ollama model size/VRAM and hardware separately. Local inference API usage cost is usually zero, but hardware/electricity are not."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2))
if __name__=="__main__": main()
