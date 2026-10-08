from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_engine.findings import run_all


# ============================================================
# 1. PLANTED RISKS
# ============================================================

PLANTED_RISKS = {
    "cash_conversion": {
        "finding_id": "cash_conversion",
        "description": "Revenue growth while operating cash flow declines",
    },

    "customer_concentration": {
        "finding_id": "cust_conc",
        "description": "Customer concentration",
    },

    "customer_contract_expiry": {
        "finding_prefix": "renewal_",
        "description": "Major customer contract expires soon",
    },

    "supplier_concentration": {
        "finding_id": "sup_conc",
        "description": "Supplier concentration",
    },

    "debt_maturity": {
        "finding_id": "debt_maturity",
        "description": "Debt maturity exposure",
    },

    "threshold_clustering": {
        "finding_prefix": "threshold_",
        "description": "Payments clustered below approval threshold",
    },
}


# ============================================================
# 2. HELPERS
# ============================================================

def get_finding_id(finding):
    """Safely extract finding ID."""
    return str(finding.get("id", ""))


def has_evidence(finding):
    """
    Check whether a finding contains meaningful supporting evidence.

    We intentionally inspect several common evidence fields because
    different risk-engine findings may store evidence differently.
    """

    evidence_fields = [
        "evidence",
        "supporting_evidence",
        "evidence_text",
        "details",
        "explanation",
        "reason",
        "description",
    ]

    for field in evidence_fields:
        value = finding.get(field)

        if value is None:
            continue

        # String evidence
        if isinstance(value, str) and value.strip():
            return True

        # List/dict evidence
        if isinstance(value, (list, dict)) and len(value) > 0:
            return True

    return False


def matches_risk(finding, risk_config):
    """Determine whether a finding corresponds to a planted risk."""

    finding_id = get_finding_id(finding)

    if "finding_id" in risk_config:
        return finding_id == risk_config["finding_id"]

    if "finding_prefix" in risk_config:
        return finding_id.startswith(risk_config["finding_prefix"])

    return False


# ============================================================
# 3. EVALUATION
# ============================================================

def evaluate():

    start = time.perf_counter()

    result = run_all("data/raw")

    runtime = time.perf_counter() - start

    findings = result["findings"]

    # --------------------------------------------------------
    # Risk detection
    # --------------------------------------------------------

    risk_results = {}

    for risk_name, risk_config in PLANTED_RISKS.items():

        matching_findings = [
            finding
            for finding in findings
            if matches_risk(finding, risk_config)
        ]

        detected = len(matching_findings) > 0

        evidence_found = any(
            has_evidence(finding)
            for finding in matching_findings
        )

        risk_results[risk_name] = {
            "detected": detected,
            "evidence": evidence_found,
            "findings": matching_findings,
        }

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    total_risks = len(risk_results)

    detected_risks = sum(
        result["detected"]
        for result in risk_results.values()
    )

    risk_recall = (
        detected_risks / total_risks
        if total_risks
        else 0
    )

    # --------------------------------------------------------
    # Evidence coverage
    # --------------------------------------------------------

    risks_with_evidence = sum(
        result["evidence"]
        for result in risk_results.values()
        if result["detected"]
    )

    evidence_coverage = (
        risks_with_evidence / detected_risks
        if detected_risks
        else 0
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 65)
    print("DEALLENS SYNTHETIC RISK EVALUATION")
    print("=" * 65)

    print(f"\nPlanted risk checks : {total_risks}")
    print(f"Detected correctly  : {detected_risks}")
    print(f"Risk recall         : {risk_recall:.2%}")

    print(
        f"Evidence coverage   : "
        f"{risks_with_evidence}/{detected_risks} "
        f"({evidence_coverage:.2%})"
    )

    print(f"Runtime             : {runtime:.3f} seconds")
    print(f"Total findings      : {len(findings)}")

    # --------------------------------------------------------
    # Detailed evaluation
    # --------------------------------------------------------

    print("\nDetailed results:")
    print("-" * 65)

    for risk_name, result in risk_results.items():

        detection_status = (
            "PASS" if result["detected"]
            else "MISS"
        )

        evidence_status = (
            "EVIDENCE"
            if result["evidence"]
            else "NO EVIDENCE"
        )

        print(
            f"{detection_status:5} | "
            f"{evidence_status:10} | "
            f"{risk_name}"
        )

    # --------------------------------------------------------
    # Finding details
    # --------------------------------------------------------

    print("\nDetected finding IDs:")
    print("-" * 65)

    for finding in findings:

        finding_id = finding.get("id")
        severity = finding.get("severity")
        title = finding.get("title")

        print(
            f"{str(finding_id):25} | "
            f"{str(severity):6} | "
            f"{title}"
        )

    # --------------------------------------------------------
    # Evidence details
    # --------------------------------------------------------

    print("\nEvidence verification:")
    print("-" * 65)

    for risk_name, result in risk_results.items():

        if not result["detected"]:
            print(f"MISS     | {risk_name}")
            continue

        if result["evidence"]:
            print(f"VERIFIED | {risk_name}")
        else:
            print(f"NO DATA  | {risk_name}")

    print("\n" + "=" * 65)

    return {
        "total_risks": total_risks,
        "detected_risks": detected_risks,
        "risk_recall": risk_recall,
        "risks_with_evidence": risks_with_evidence,
        "evidence_coverage": evidence_coverage,
        "runtime": runtime,
        "findings": findings,
        "risk_results": risk_results,
    }


if __name__ == "__main__":
    evaluate()