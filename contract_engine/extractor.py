"""STEP 7 - Contract intelligence.
(a) Structured contracts.csv path (works offline).
(b) Text path: regex first-pass + optional Claude extraction with verbatim quote spans."""
import re, json, os
import pandas as pd

CLAUSE_PATTERNS = {
    "change_of_control": r"(change of control|change in control|assign(ment)? .{0,40}consent)",
    "termination_convenience": r"terminat\w+ .{0,60}(for convenience|without cause)",
    "exclusivity": r"\b(exclusiv\w+|sole supplier|non-compete)\b",
    "auto_renewal": r"(auto(matic(ally)?)?[- ]renew\w*|renews? automatically)",
    "uncapped_liability": r"(unlimited liability|uncapped)",
    "penalty": r"(liquidated damages|penalt(y|ies))",
}


def regex_clause_scan(text: str) -> list[dict]:
    """Returns found clause types with a short verbatim quote (evidence-or-nothing)."""
    hits = []
    for name, pat in CLAUSE_PATTERNS.items():
        m = re.search(pat, text, flags=re.I)
        if m:
            s, e = max(0, m.start() - 80), min(len(text), m.end() + 120)
            hits.append({"clause": name, "quote": text[s:e].replace("\n", " ").strip()})
    return hits


def llm_extract(text: str, model: str = os.getenv("DEALLENS_MODEL", "claude-sonnet-5")) -> dict | None:
    """Optional: needs ANTHROPIC_API_KEY. Model must return null when not found and quote every field."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    client = anthropic.Anthropic()
    prompt = ("Extract these fields from the contract as JSON only: counterparty, contract_value, start_date, "
              "end_date, renewal_terms, termination_clause, payment_terms, penalty_clauses, exclusivity, "
              "change_of_control. For EACH field return {\"value\":..., \"quote\":\"verbatim text\"}. "
              "If a field is not present return null. Never guess.\n\nCONTRACT:\n" + text[:60000])
    r = client.messages.create(model=model, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
    raw = r.content[0].text.strip().strip("`").removeprefix("json").strip()
    return json.loads(raw)


def contract_findings(contracts: pd.DataFrame, cust_shares: dict, horizon_days: int = 180) -> list[dict]:
    """CROSS-DOCUMENT reasoning: revenue share (customers.csv) x contract expiry (contracts.csv)."""
    out = []
    today = pd.Timestamp.today().normalize()
    c = contracts.copy()
    c["end_date"] = pd.to_datetime(c["end_date"])
    for _, r in c.iterrows():
        days = (r["end_date"] - today).days
        share = cust_shares.get(r["counterparty"], 0.0) if r["type"] == "customer" else 0.0
        if r["type"] == "customer" and days <= horizon_days and share >= 0.10:
            out.append({
                "id": f"renewal_{r['contract_id']}", "category": "Contract exposure",
                "severity": "High" if share >= 0.20 else "Medium",
                "title": f"Major customer contract expires in {days} days ({r['counterparty']})",
                "detail": f"{r['counterparty']} = {share:.0%} of revenue; contract {r['contract_id']} ends "
                          f"{r['end_date'].date()}.",
                "evidence": [{"source": "contracts.csv", "contract_id": r["contract_id"]},
                             {"source": "customers.csv", "field": "revenue"}],
                "questions": ["Renewal status and negotiation history?", "Any pricing concessions requested?"],
            })
        if bool(r.get("change_of_control", False)):
            out.append({
                "id": f"coc_{r['contract_id']}", "category": "Contract exposure", "severity": "Medium",
                "title": f"Change-of-control clause ({r['counterparty']}, {r['contract_id']})",
                "detail": f"The acquisition may trigger termination or consent rights"
                          f"{f' covering {share:.0%} of revenue' if share else ''}.",
                "evidence": [{"source": "contracts.csv", "contract_id": r["contract_id"]}],
                "questions": ["Obtain counterparty consent or waiver as a condition precedent?"],
            })
    return out
