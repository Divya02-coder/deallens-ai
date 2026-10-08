import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

from risk_engine.findings import run_all


BENCHMARK = ROOT / "evaluation" / "benchmark"

DATA_DIR = BENCHMARK / "data_holdout"

GROUND_TRUTH = BENCHMARK / "ground_truth_holdout.json"


RISK_ID_PREFIXES = {
    "cash_conversion": [
        "cash_conversion",
    ],

    "customer_concentration": [
        "cust_conc",
    ],

    "supplier_concentration": [
        "sup_conc",
    ],

    "debt_maturity": [
        "debt_maturity",
    ],

    "contract_expiry": [
        "renewal_",
    ],

    "threshold_clustering": [
        "threshold_",
    ],
}


def finding_to_risk_type(finding_id):

    for risk_type, prefixes in RISK_ID_PREFIXES.items():

        for prefix in prefixes:

            if finding_id == prefix or finding_id.startswith(prefix):
                return risk_type

    return None


def detected_risk_types(findings):

    detected = set()

    for finding in findings:

        risk_type = finding_to_risk_type(
            finding.get("id", "")
        )

        if risk_type:
            detected.add(risk_type)

    return detected


def evidence_coverage(findings):

    relevant = []

    for finding in findings:

        risk_type = finding_to_risk_type(
            finding.get("id", "")
        )

        if risk_type:
            relevant.append(finding)

    if not relevant:
        return None

    supported = 0

    for finding in relevant:

        evidence = finding.get("evidence", [])

        if isinstance(evidence, list) and len(evidence) > 0:
            supported += 1

    return supported / len(relevant)


def main():

    with open(GROUND_TRUTH, "r") as f:
        ground_truth = json.load(f)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    total_runtime = 0.0

    evidence_scores = []

    failures = []

    for case in ground_truth:

        case_id = case["case_id"]

        expected = case["risk_type"]

        expected_set = (
            {expected}
            if expected
            else set()
        )

        case_dir = DATA_DIR / case_id

        start = time.perf_counter()

        result = run_all(
            data_dir=str(case_dir)
        )

        runtime = time.perf_counter() - start

        total_runtime += runtime

        detected = detected_risk_types(
            result["findings"]
        )

        # Expected positive risk
        if expected:

            if expected in detected:
                tp += 1
            else:
                fn += 1

                failures.append({
                    "type": "FN",
                    "case_id": case_id,
                    "expected": expected,
                    "detected": sorted(detected),
                })

            # Unexpected additional risk
            unexpected = detected - expected_set

            fp += len(unexpected)

            for risk in sorted(unexpected):

                failures.append({
                    "type": "FP",
                    "case_id": case_id,
                    "expected": expected,
                    "unexpected": risk,
                })

        # Clean case
        else:

            if detected:
                fp += len(detected)

                for risk in sorted(detected):

                    failures.append({
                        "type": "FP",
                        "case_id": case_id,
                        "expected": None,
                        "unexpected": risk,
                    })

            else:
                tn += 1

        coverage = evidence_coverage(
            result["findings"]
        )

        if coverage is not None:
            evidence_scores.append(
                coverage
            )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall)
        else 0
    )

    total_cases = len(ground_truth)

    accuracy = (
        (tp + tn) / total_cases
        if total_cases
        else 0
    )

    avg_runtime = (
        total_runtime / total_cases
        if total_cases
        else 0
    )

    evidence = (
        sum(evidence_scores) /
        len(evidence_scores)
        if evidence_scores
        else 0
    )

    print()
    print("=" * 65)
    print("DEALLENS INDEPENDENT HOLDOUT BENCHMARK")
    print("=" * 65)

    print()
    print(f"Cases evaluated : {total_cases}")
    print(f"True positives  : {tp}")
    print(f"False positives : {fp}")
    print(f"False negatives : {fn}")
    print(f"True negatives  : {tn}")

    print()
    print("Metrics:")
    print("-" * 65)

    print(
        f"Precision       : {precision * 100:.2f}%"
    )

    print(
        f"Recall          : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score        : {f1 * 100:.2f}%"
    )

    print(
        f"Accuracy        : {accuracy * 100:.2f}%"
    )

    print(
        f"Evidence coverage : {evidence * 100:.2f}%"
    )

    print(
        f"Avg runtime     : {avg_runtime:.3f}s"
    )

    print()
    print("Failure analysis:")
    print("-" * 65)

    if failures:

        for failure in failures:

            if failure["type"] == "FN":

                print(
                    f"FN | {failure['case_id']:<12} | "
                    f"expected={failure['expected']} | "
                    f"detected={failure['detected']}"
                )

            else:

                print(
                    f"FP | {failure['case_id']:<12} | "
                    f"expected={failure.get('expected')} | "
                    f"unexpected={failure.get('unexpected')}"
                )

    else:

        print(
            "No false positives or false negatives."
        )

    results = {
        "benchmark": "independent_holdout",
        "cases": total_cases,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "evidence_coverage": evidence,
        "avg_runtime_seconds": avg_runtime,
        "failures": failures,
    }

    output = (
        BENCHMARK /
        "holdout_results.json"
    )

    with open(output, "w") as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("Results saved to:")
    print(output)


if __name__ == "__main__":
    main()