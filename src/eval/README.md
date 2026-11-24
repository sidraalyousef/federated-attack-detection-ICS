# Evaluation (`src/eval`)

All experiments score models through `evaluate()`, so every table uses the same protocols.

## Usage

```python
from src.eval import evaluate, threshold_from_normal, window_scores_to_points, summary_row

# 1. Window scores -> per-timestep scores (score goes to each window's last point)
# (windows cut with stride=1, so stride is left at its default of 1)
test_pts = window_scores_to_points(test_window_errors, n_points=len(test_labels), window_size=W)
val_pts  = window_scores_to_points(val_window_errors,  n_points=len(val),         window_size=W)

# 2. Threshold from NORMAL validation data only (no test labels)
thr = threshold_from_normal(val_pts, method="percentile", q=99)

# 3. Evaluate (NaN warm-up points are dropped automatically)
res = evaluate(test_pts, test_labels, thr)
print(summary_row(res, "LSTM-AE centralized"))
```

Per-client results (federated): `evaluate_groups(scores, labels, client_ids, thr)`.
`thr` can be one global float or a `{client: threshold}` dict.

## Rules for the team

1. **Report all three F1s:** `raw_f1` (headline), `pa_f1` (for comparison with papers only), and `pak_auc_f1`. Add `auroc`/`auprc` as threshold-free numbers, and `etapr_f1` (computed by default) on every dataset. It's the HAI authors' own metric, so HAI tables must include it.
2. **Threshold:** `threshold_from_normal` on a held-out slice of normal training data, with the same rule (default: 99th percentile) for every model. `best_f1_threshold` uses test labels. It is an oracle upper bound and must be labelled as one.
3. **Window alignment:** `mode="last"` by default. Labels are never windowed: keep the original per-timestep labels, and let `evaluate` drop the NaN warm-up points. **Score validation and test windows with stride 1** (`EVAL_STRIDE` in `src/data_loaders/preprocessing.py`). Training windows can use stride 10, but with `mode="last"` a stride of 10 leaves 90% of the seconds without a score, and `window_scores_to_points` warns about it.
4. **Sanity check:** every results table includes a `random_scores` row. A model that doesn't clearly beat it on `raw_f1` and `pak_auc_f1` isn't detecting anything.

PA%K follows Kim et al., AAAI-22: a segment is filled in when more than K% of its points are flagged. K=0 is PA and K=100 is raw. The default sweep is K = 0, 10, …, 100, and the reported number is the area under that sweep.

eTaPR (enhanced time-series aware precision/recall, Hwang et al. 2022) comes from the `faster-etapr` package, pinned in `requirements.txt`. It counts an attack as detected if at least 10% of it is covered (`theta_r=0.1`), and an alarm as correct if at least 50% of it overlaps an attack (`theta_p=0.5`). These are the authors' defaults. Keep them fixed for all reported results. Call it directly with `etapr_scores(preds, labels)`, or switch it off with `evaluate(..., include_etapr=False)`.

## Federated results (`src/eval/federated.py`)

These helpers only arrange results that `evaluate` already produced; they don't compute new metrics.

```python
from src.eval import evaluate_groups
from src.eval.federated import client_table, client_summary, compare_settings, pooled_evaluate

per_client = evaluate_groups(scores, labels, client_ids, thresholds)   # {client: result}
table   = client_table(per_client)        # one row per client
summary = client_summary(table)           # mean / std / min across clients

compare_settings({"centralized": res_central, "local-only": res_local, "FedAvg": res_fedavg})
pooled = pooled_evaluate(scores, labels, client_ids, thresholds)       # all clients as one test set
```

Rules for federated tables:

1. **Same data for every setting.** Score the centralized model on the same client slices as the federated models, so the rows of a comparison table are comparable.
2. **Clients without attacks.** Every F1 is 0 and AUROC/AUPRC are NaN on such a client, whatever the model does. The tables set those metrics to NaN so means skip them, and keep `n_false_alarms` and `fpr`. Judge a no-attack client by its false alarms.
3. **Per-client mean vs pooled.** The per-client mean treats clients equally, while `pooled_evaluate` weighs clients by their number of attack points. Say which one a table shows. With per-client thresholds, `pooled_evaluate` scores each client's margin over its own threshold.
4. **Windows are built inside each client**, after the split, so no window crosses a client boundary. Split between attacks so each client holds whole attacks. `pooled_evaluate` warns if an attack touches a client boundary.

`evaluate` also returns `n_false_alarms` and `fpr` (false alarms divided by the number of normal timesteps).

Tests: `python -m pytest tests -q`

## By-stage (vertical) results (`src/eval/vertical.py`)

In the by-stage split (`split_clients_by_stage`) every client covers the **same seconds** with only its own sensors, so the helpers above don't fit: pooling stage clients would count each attack second once per client. Combine the stages per second instead:

```python
from src.eval import evaluate, threshold_from_normal
from src.eval.federated import client_table
from src.eval.vertical import (points_from_stages, evaluate_stages,
                               evaluate_own_attacks, evaluate_any_alarm)

test = points_from_stages(test_window_scores, clients)               # {"S1": per-second, ...}
val  = points_from_stages(val_window_scores, clients, part="val")
thr  = {k: threshold_from_normal(v) for k, v in val.items()}         # one threshold per stage
labels = clients[0]["attack_labels"]

per_stage = evaluate_stages(test, labels, thr)                        # 1. each stage vs all attacks
own       = evaluate_own_attacks(test, labels, thr, attack_stages)    # 2. each stage vs its own attacks
plant     = evaluate_any_alarm(test, labels, thr)                     # 3. alarm if ANY stage alarms

central = evaluate(central_points, labels, central_thr)              # centralized: full attack file
client_table({"centralized": central, "local any-alarm": plant})     # one row per setting
```

Rules:

1. **The plant-wide number is `evaluate_any_alarm`.** Put it next to the centralized and federated models. Per-stage results (`evaluate_stages`) are expected to have low recall, since a stage can't see attacks elsewhere. Don't average them into a plant score.
2. **`attack_stages`** lists the targeted stage(s) for each attack segment in the labels, in the order of `get_segments(labels)`, e.g. `[(1,), (1,), (3,), (1, 3), ...]`. Build it once from the official attack list. There is one entry per *labelled* segment, which isn't always one per listed attack. Use `()` when the target is unknown. Attacks on other stages are left out of a stage's score, so they count neither as misses nor as false alarms.
3. **Any-alarm uses `mode="ratio"` by default:** each stage's score is divided by its own threshold, so stages with large error scales don't dominate AUROC/AUPRC. The alarms are the same in either mode. `stage_alarms` and `stage_false_alarms` show which stage causes the false alarms.
