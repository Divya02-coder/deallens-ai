from evaluation.metrics import score

def test_category_metrics():
    rows=score([
        {"expected_categories":["a","b"],"predicted_categories":["a"]},
        {"expected_categories":["b"],"predicted_categories":["b","c"]},
    ])
    by={r["category"]:r for r in rows}
    assert by["a"]["precision"]==1.0 and by["a"]["recall"]==1.0
    assert by["b"]["precision"]==1.0 and by["b"]["recall"]==0.5
    assert by["c"]["precision"]==0.0 and by["c"]["recall"]==0.0
