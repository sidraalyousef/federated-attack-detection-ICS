# Preprocessing (`src/data_loaders`)

`preprocessing.py` turns the raw SWaT CSVs into scaled, windowed arrays that the models train and test on.

## What it does

1. Loads `SWaT_Dataset_Normal_v1.csv` and `SWaT_Dataset_Attack_v0.csv`, fixing the banner row above the header and the "A ttack" label typos.
2. **Drops the first 6 hours of the normal file** (21,600 s, `WARMUP_S`). The plant is still settling after start-up, and most SWaT papers remove this part.
3. **Drops 8 sensors:**
   - 6 that never change in either file: P202, P401, P404, P502, P601, P603.
   - **P201** and **P403** (`EXCLUDE`). Both are constant in the normal file but change during *normal* periods of the attack file (P201 for 53,732 normal seconds vs 739 attack seconds; P403 for 60 normal seconds and never during attacks), so they would mostly raise false alarms.
   - Sensors that are constant in the normal file but change (almost) only during attacks are **kept**: P102 (3,029 attack seconds, 96 normal), P204 and P206 (attack seconds only). This choice was made after looking at the attack file; say so in the report.
4. **Splits the normal data in time** (`VAL_FRAC`): the first 85% is the training set, the last 15% (27 Dec 14:16 → 28 Dec 10:00) is the validation set. The validation set is never trained on; use it to pick the threshold (`src.eval.threshold_from_normal`).
5. **Min-max scales every sensor using the training part only.** Validation and attack values can therefore go outside 0–1, which is what lets attacks stand out.

The result is 2-D: one row per second, 43 sensors. Labels stay as one 0/1 value per second of the attack file.

### Windows

Windows are cut afterwards with `make_windows` (60-second windows):

- **Training:** stride 10 (`TRAIN_STRIDE`), one window every 10 s.
- **Validation and attack (anything you score): stride 1** (`EVAL_STRIDE`). With stride 10, only every 10th second would get a score and the evaluation would silently drop the rest.

`make_windows` returns a read-only view that shares memory with the array, so it costs nothing to build. Copy each batch before giving it to PyTorch (`torch.tensor(batch)`, not `torch.from_numpy`). Conv1d layers expect `(batch, sensors, time)`, so transpose with `.transpose(0, 2, 1)` / `.permute(0, 2, 1)`.

## Before you run it

Put the two raw files directly in `data/raw/swat/`, with these exact names:

```
data/raw/swat/
├── SWaT_Dataset_Normal_v1.csv
└── SWaT_Dataset_Attack_v0.csv
```

Activate your virtual environment (see the main README) and run everything from the **repo root**, not from inside `src/`.

## How to run it

Run this once from the repo root. It works as-is in PowerShell and in a Linux/macOS terminal:

```bash
python -c "from src.data_loaders.preprocessing import prepare_swat; import numpy as np; d=prepare_swat('data/raw/swat'); np.savez('data/processed/swat_scaled.npz', train=d['train'], val=d['val'], attack=d['attack'], labels=d['attack_labels'], features=np.array(d['features']))"
```

It takes a minute or so and prints nothing when it finishes. Afterwards you should have `data/processed/swat_scaled.npz` (about 160 MB). The windows are **not** saved: saved windows would be over 4 GB, and cutting them from the 2-D arrays is instant.

## What you get

| Key | Shape | Use |
|---|---|---|
| `train` | `(402390, 43)` | normal data, 22 Dec 22:30 → 27 Dec 14:16: **train** on these |
| `val` | `(71010, 43)` | normal data, 27 Dec 14:16 → 28 Dec 10:00: **threshold** from these |
| `attack` | `(449919, 43)` | the attack file: **test** on these |
| `labels` | `(449919,)` | 0/1 attack label for each second of the attack file |
| `features` | `(43,)` | sensor names, in column order |

Load it and cut windows in your own code with:

```python
import numpy as np
from src.data_loaders.preprocessing import make_windows, EVAL_STRIDE

d = np.load("data/processed/swat_scaled.npz")
train_w = make_windows(d["train"])                          # (40234, 60, 43), stride 10
val_w   = make_windows(d["val"], stride=EVAL_STRIDE)        # (70951, 60, 43)
test_w  = make_windows(d["attack"], stride=EVAL_STRIDE)     # (449860, 60, 43)
labels  = d["labels"]
```

## Using the function directly

If you need more than the saved arrays, call `prepare_swat` yourself:

```python
from src.data_loaders.preprocessing import prepare_swat

d = prepare_swat("data/raw/swat")                         # defaults: 6 h warm-up, 15% validation
d = prepare_swat("data/raw/swat", val_frac=0.2)            # other settings
```

Besides the arrays and labels, the returned dict has:

- `features`: names of the 43 sensors kept, in column order
- `dropped`: all removed sensors; `dropped_constant` and `excluded` say why
- `scaler`: `(min, range)` fitted on the training part, to scale new data the same way
- `train_timestamps`, `val_timestamps`, `attack_timestamps`: the time of each row
- `warmup`, `val_frac`: the settings used

## Client split (`clients.py`)

For local-only and federated training, `split_clients` splits the output of `prepare_swat` **by time**:

- **Test data:** the attack file is cut only in quiet periods, so every attack stays whole and is followed by at least 1 hour of recovery (`MIN_QUIET_S`) inside the same client. `split_clients` checks this and raises an error if a cut breaks it.
- **Training and validation data:** both are cut into equal consecutive pieces; client k gets piece k of each. Joined back together, the pieces are exactly the centralized train and validation sets, so centralized, local-only and FedAvg all use the same data.

**5 clients (main setting).** C4 has no attacks: judge it by its false alarms, not F1.

| Client | Test period | Test | Attacks | Attack seconds | Train / val |
|---|---|---|---|---|---|
| C1 | 28 Dec 10:00 → 29 Dec 13:02 | 27.0 h | 12 | 6,127 | 22.4 h / 3.9 h |
| C2 | 29 Dec 13:02 → 30 Dec 18:29 | 29.4 h | 9 | 5,630 | 22.4 h / 3.9 h |
| C3 | 30 Dec 18:29 → 31 Dec 23:11 | 28.7 h | 4 | 37,559 (incl. attack #27) | 22.4 h / 3.9 h |
| C4 | 31 Dec 23:11 → 1 Jan 10:25 | 11.2 h | 0 | 0 | 22.4 h / 3.9 h |
| C5 | 1 Jan 10:26 → 2 Jan 14:59 | 28.6 h | 10 | 5,305 | 22.4 h / 3.9 h |

**3 clients (check on the effect of client count).** C1+C2, C3+C4 and C5 of the 5-client split, renamed C1–C3: 56.5 h / 39.9 h / 28.6 h of test data with 21 / 4 / 10 attacks. Every client has attacks here; the no-attack rule applies to the 5-client setting only, because a no-attack client can be at most about 16 h long and would leave the other two very uneven.

```python
from src.data_loaders.preprocessing import prepare_swat
from src.data_loaders.clients import split_clients, client_windows, points_from_clients, summary

d = prepare_swat("data/raw/swat")
clients = split_clients(d, n_clients=5)      # or 3
print(summary(clients))

for c in clients:
    w = client_windows(c)    # windows cut inside the client: train stride 10, val/attack stride 1
    ...                      # train on w["train"], threshold from w["val"], score w["attack"]

# test_scores[k] = window scores for client k's attack windows
scores, labels, groups = points_from_clients(test_scores, clients)
# -> src.eval.evaluate_groups(scores, labels, groups, thresholds)
```

**Centralized model:** train on `make_windows(d["train"])` and threshold on `make_windows(d["val"], stride=1)` as above, but **score it on each client's attack windows** (`client_windows(c)["attack"]`), so its results line up with the federated ones row for row.

## Checking the output

`notebooks/02_swat_preprocessing.ipynb` runs the same function and plots a few sensors over the attack file. It saves nothing, so it's safe to run any time to see what the data looks like.

## Things to know

- **One attack dominates the labels.** Attack #27 is labelled as lasting about 10 hours, but the official attack list gives it about 28 minutes. That single block is 66% of all attack seconds, so per-second scores mostly reflect it. Report per-attack detection (eTaPR, see `src/eval`) alongside point-wise F1.
- **Some analyser sensors drift** (AIT201, AIT203, AIT402, AIT502). They sit outside the 0–1 range even in normal seconds of the attack file, which can cause false alarms. They are kept for now; the plan is to run the centralized baseline with and without them.
- **The output is not committed.** `data/processed/` is gitignored, so everyone runs this step on their own machine.
