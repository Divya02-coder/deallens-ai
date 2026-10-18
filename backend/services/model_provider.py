"""Optional local Ollama-compatible open-weight model adapter."""
from __future__ import annotations
import json, os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

def generate_local(prompt: str, system_instruction: str="", model: str|None=None,
                   timeout: int=120) -> dict:
    """Call a local Ollama server. Returns text, latency_ms, model and provider."""
    import time
    base=os.getenv("OLLAMA_BASE_URL","http://localhost:11434").rstrip("/")
    model=model or os.getenv("OLLAMA_MODEL","qwen2.5:7b")
    body={"model":model,"prompt":prompt,"system":system_instruction,"stream":False,
          "options":{"temperature":0.2}}
    req=Request(base+"/api/generate",data=json.dumps(body).encode(),
                headers={"Content-Type":"application/json"},method="POST")
    start=time.perf_counter()
    try:
        with urlopen(req,timeout=timeout) as response:
            data=json.loads(response.read().decode())
    except (URLError,HTTPError,TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed at {base}: {exc}") from exc
    return {"text":data.get("response","").strip(),"latency_ms":round((time.perf_counter()-start)*1000,2),
            "model":data.get("model",model),"provider":"ollama",
            "prompt_eval_count":data.get("prompt_eval_count"),"eval_count":data.get("eval_count")}
