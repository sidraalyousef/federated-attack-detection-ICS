# Preprocessing (`src/data_loaders`)

`preprocessing.py` turns the raw SWaT CSVs into scaled, windowed arrays that the models train and test on.

## What it does

1. Loads `SWaT_Dataset_Normal_v1.csv` and `SWaT_Dataset_Attack_v0.csv`, fixing the banner row above the header and the "A ttack" label typos.
2. Drops the sensors that never change in *either* file (6 of them). Sensors that are constant in the normal file but move during attacks (e.g. pump P102) are kept.
3. Min-max scales every sensor using the **normal file only**. Attack-file values can therefore go outside 0–1, which is what lets attacks stand out.
4. Cuts both files into **60-second windows**, one starting every **10 seconds**.

Labels are **not** windowed: they stay as one 0/1 value per second of the attack file. `src/eval` lines window scores back up with them.

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
python -c "from src.data_loaders.preprocessing import prepare_swat; import numpy as np; d=prepare_swat('data/raw/swat'); np.savez('data/processed/swat_windows.npz', normal=d['normal_windows'], attack=d['attack_windows'], labels=d['attack_labels'])"
```

It takes a minute or so and prints nothing when it finishes. Afterwards you should have `data/processed/swat_windows.npz`.

## What you get

| Key | Shape | Use |
|---|---|---|
| `normal` | `(49495, 60, 45)` | windows from the normal file: **train** on these |
| `attack` | `(44986, 60, 45)` | windows from the attack file: **test** on these |
| `labels` | `(449919,)` | 0/1 attack label for each second of the attack file |

Shapes are `(number of windows, 60 seconds, 45 sensors)`.

Load it in your own code with:

```python
import numpy as np

d = np.load("data/processed/swat_windows.npz")
normal, attack, labels = d["normal"], d["attack"], d["labels"]
```

## Using the function directly

If you need more than the three saved arrays, call `prepare_swat` yourself:

```python
from src.data_loaders.preprocessing import prepare_swat

d = prepare_swat("data/raw/swat")            # defaults: window=60, stride=10
d = prepare_swat("data/raw/swat", window=120, stride=20)   # other settings
```

Besides the windows and labels, the returned dict has:

- `features`: names of the 45 sensors kept, in column order
- `dropped`: names of the sensors removed
- `scaler`: `(min, range)` fitted on the normal file, to scale new data the same way
- `normal_timestamps`, `attack_timestamps`: the time of each row
- `window`, `stride`: the settings used

## Checking the output

`notebooks/02_swat_preprocessing.ipynb` runs the same function and plots a few sensors over the attack file. It saves nothing, so it's safe to run any time to see what the data looks like.

## Things to know

- **One attack dominates the labels.** Attack #27 is labelled as lasting about 10 hours, but the official attack list gives it about 28 minutes. That single block is 66% of all attack seconds, so per-second scores mostly reflect it. Report per-attack detection (eTaPR, see `src/eval`) alongside point-wise F1.
- **Three sensors drift** (AIT201, AIT402, AIT502). They sit outside the 0–1 range even in normal seconds of the attack file, which can cause false alarms.
- **The output is not committed.** `data/processed/` is gitignored, so everyone runs this step on their own machine.
