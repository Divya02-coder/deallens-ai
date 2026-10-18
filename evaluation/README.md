# Evaluation protocol

## Risk-engine regression benchmark
Run `python evaluation/benchmark/evaluate_benchmark.py` for the repository's deterministic synthetic holdout. Treat these cases as regression tests, not evidence of real-world 100% accuracy.

## Labelled category evaluation
Create a JSONL file with one line per reviewed case:

```json
{"case_id":"contract-001","expected_categories":["change_of_control","termination"],"predicted_categories":["change_of_control"],"citation_correct":true}
```

Then run:

```bash
python evaluation/metrics.py evaluation/labelled_cases.jsonl --output evaluation/results.json
```

The script reports per-category precision, recall and F1. Keep training/dev/test cases separate and report test-set size.

## CUAD
CUAD is a contract-clause review dataset, not a full M&A financial-risk benchmark. Download it separately from its official project and follow its license/terms. For a valid experiment, map the CUAD question/answer annotations to clause categories, run the contract extractor on the corresponding documents, and manually validate the mapping before computing category metrics. Do not report synthetic-data metrics as CUAD results. Citation correctness should be separately measured by checking whether the cited document/page/span actually contains the claim.
