# Evaluation (`src/eval`)

All experiments score models through `evaluate()`, so every table uses the same protocols.

## Usage

```python
from src.eval import evaluate, threshold_from_normal, window_scores_to_points, summary_row

# 1. Window scores -> per-timestep scores (score goes to each window's last point)
test_pts = window_scores_to_points(test_window_errors, n_points=len(test_labels), window_size=W)
val_pts  = window_scores_to_points(val_window_errors,  n_points=len(val_df),      window_size=W)

# 2. Threshold from NORMAL validation data only (no test labels)
thr = threshold_from_normal(val_pts, method="percentile", q=99)

# 3. Evaluate (NaN warm-up points are dropped automatically)
res = evaluate(test_pts, test_labels, thr)
print(summary_row(res, "LSTM-AE centralized"))
```

Per-client results (federated): `evaluate_groups(scores, labels, client_ids, thr)`.
`thr` can be one global float or a `{client: threshold}` dict.

## Rules for the team

1. **Report all three F1s:** `raw_f1` (headline), `pa_f1` (for comparison with papers only), and `pak_auc_f1`. Add `auroc`/`auprc` as threshold-free numbers.
2. **Threshold:** `threshold_from_normal` on a held-out slice of normal training data, with the same rule (default: 99th percentile) for every model. `best_f1_threshold` uses test labels. It is an oracle upper bound and must be labelled as one.
3. **Window alignment:** `mode="last"` by default. Labels are never windowed: keep the original per-timestep labels, and let `evaluate` drop the NaN warm-up points.
4. **Sanity check:** every results table includes a `random_scores` row. A model that doesn't clearly beat it on `raw_f1` and `pak_auc_f1` isn't detecting anything.

PA%K follows Kim et al., AAAI-22: a segment is filled in when more than K% of its points are flagged. K=0 is PA and K=100 is raw. The default sweep is K = 0, 10, …, 100, and the reported number is the area under that sweep.

Tests: `python -m pytest tests -q`
