# Centralized LSTM-AE baseline: results

One model trained on all normal training windows, scored on each client's attack windows
(5 time-based clients), so the numbers line up row for row with the Conv-AE, local-only
and federated runs.

Reproduce (from the repo root; about 13 min on a laptop CPU):

```bash
python -m experiments.lstm_ae_centralized                       # default sensors
python -m experiments.lstm_ae_centralized --drop-drifting --out experiments/lstm_ae_centralized/outputs/drop_drifting
python -m experiments.lstm_ae_per_sensor_error                  # which sensors cause the false alarms
```

## Setup

| | |
|---|---|
| Model | `LSTMAE`: 1-layer LSTM encoder (64 hidden) -> 16-d latent -> LSTM decoder, 66,107 parameters |
| Data | SWaT 2015, 43 sensors (P201, P403 dropped), 60 s windows |
| Training | 40,234 normal windows (stride 10), Adam lr 1e-3, batch 256, 20 epochs, seed 0 |
| Score | mean squared reconstruction error of the window, mapped to seconds |
| Threshold | 99th percentile of the validation scores (normal data only, no test labels) |
| Test | each client's attack windows, stride 1 (5 clients, C4 has no attacks) |

Training loss falls from 0.0715 to 0.0020 and validation loss from 0.0394 to 0.0034 over 20 epochs, and had not
levelled off, so more epochs may help.

## Results

"Detected" counts an attack when at least one of its seconds is flagged, so it is lenient: the random
baseline gets 34/35 with the same rule. Use raw F1, eTaPR F1 and the false-alarm rate as the main numbers.

### Default sensors (43)

| Client | raw F1 | eTaPR F1 | AUROC | Attacks detected | False alarms (s) | FPR |
|---|---|---|---|---|---|---|
| C1 | 0.433 | 0.204 | 0.843 | 10/12 | 8,819 | 0.097 |
| C2 | 0.114 | 0.133 | 0.692 | 9/9 | 82,949 | 0.827 |
| C3 | 0.534 | 0.000 | 0.932 | 4/4 | 65,661 | 1.000 |
| C4 (no attacks) | n/a | n/a | n/a | 0/0 | 40,400 | 1.000 |
| C5 | 0.098 | 0.000 | 0.806 | 10/10 | 97,476 | 1.000 |
| **Mean, clients with attacks** | **0.295** | **0.084** | **0.818** | | | 0.785 |
| **Pooled** | **0.260** | **0.133** | **0.827** | 33/35 | | |

### Without the four drifting analysers (39 sensors: AIT201, AIT203, AIT402, AIT502 also dropped)

| Client | raw F1 | eTaPR F1 | AUROC | Attacks detected | False alarms (s) | FPR |
|---|---|---|---|---|---|---|
| C1 | 0.447 | 0.223 | 0.768 | 9/12 | 7,697 | 0.084 |
| C2 | 0.198 | 0.127 | 0.844 | 8/9 | 35,071 | 0.350 |
| C3 | 0.538 | 0.000 | 0.957 | 3/4 | 64,195 | 0.978 |
| C4 (no attacks) | n/a | n/a | n/a | 0/0 | 40,400 | 1.000 |
| C5 | 0.098 | 0.000 | 0.890 | 10/10 | 97,385 | 0.999 |
| **Mean, clients with attacks** | **0.320** | **0.087** | **0.865** | | | 0.682 |
| **Pooled** | **0.292** | **0.114** | **0.909** | 30/35 | | |

Random scores with the same rule: raw F1 0.019, AUROC 0.502 (pooled).

## What this shows

- **The model ranks attacks well but the alarm threshold does not carry over.** AUROC is 0.83 to 0.91 pooled, far above the
  random 0.50, so attack seconds really do get higher errors. But the threshold fixed on the validation days
  (27 to 28 Dec) flags almost every second on C3, C4 and C5 (FPR about 1.0), so F1 is low there.
- **The false alarms come from sensors that drift, not from the attacks.** C4 has no attacks and is flagged in full.
  With default sensors, AIT201 alone has 3,229 times its validation error on C4; with the four known drifting
  analysers dropped, AIT504, AIT501 and AIT202 (water-quality analysers) take over and give 79% of the error
  (`experiments/lstm_ae_per_sensor_error.py`). Dropping the first four helps C2 (FPR 0.83 -> 0.35) and raises mean raw F1
  from 0.295 to 0.320, but does not fix C3 to C5.
- **Next step for the baselines:** decide whether to drop or rescale all drifting analysers (AIT2xx, AIT4xx, AIT5xx),
  or to set a per-client or rolling threshold. Whatever is chosen should be applied to every baseline (Conv-AE,
  local-only, federated) so they stay comparable.
- **Attack #27** (labelled as about 10 hours, mostly in C3) dominates the point-wise labels (see PREPROCESSING.md),
  so C3's raw F1 of about 0.53 mostly reflects that one block.

## Limits

- Single seed (0) and a single configuration; no hyper-parameter search. Differences of a few F1 points between the
  two sensor sets should not be over-read.
- The threshold rule (99th percentile of validation) is the one used in the Conv-AE script; it was not tuned.
- The Conv-AE numbers were not re-run here, so there is no side-by-side table yet.
