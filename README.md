# Federated Attack Detection on Industrial Control Systems (ICS)

Graduation project (ENS491/492) — federated learning for cyberattack / anomaly detection on
industrial control system infrastructure, using the **SWaT**, **WADI**, and **HAI** testbed
datasets.

**Team:** Sedra Alyousef (Sid), Salma Tubail, Fowzi Hindi

## Project overview

The project spans two semesters:

- **Semester 1 — Design phase** (9 weeks): data exploration, methodology
  selection, proposal.
- **Semester 2 — Implementation phase** (14 weeks): model implementation, federated setup,
  evaluation, and a **live running demo** (not just offline results).

### Approach

- **Models:** LSTM and convolutional autoencoders (chosen over heavier graph-based approaches
  like GDN/MTAD-GAT for tractability within the project timeline).
- **Datasets:** SWaT and WADI as the primary training/evaluation sets, with HAI used to test
  generalization across a different testbed.
- **Federated setup:** clients train locally on partitioned data; a server aggregates model
  updates (FedAvg-style, to be refined during implementation) without centralizing raw sensor
  data.

See [`docs/literature-review.md`](docs/literature-review.md) for the paper summaries and
related-work framing this project builds on.

## Repository structure

```
├── data/
│   ├── raw/             # SWaT/WADI/HAI raw data (NOT committed — see Data access below)
│   │   ├── swat/
│   │   ├── wadi/
│   │   └── hai/
│   └── processed/       # cleaned/windowed data (NOT committed)
├── notebooks/           # exploratory analysis, per-dataset EDA
├── src/
│   ├── __init__.py       
│   ├── models/          # LSTM, conv-autoencoder model definitions
│   ├── federated/       # client/server logic, aggregation strategies
│   ├── data_loaders/    # dataset loading, preprocessing, windowing
│   └── eval/            # metrics (including PA / PA%K scoring per evaluation-rigor lit)
│       ├── __init__.py   
│       ├── metrics.py   
│       └── README.md    
├── tests/               # unit tests, mirrors src/
│   ├── __init__.py       
│   └── test_metrics.py  
├── experiments/         # run configs, scripts, results tracking
├── docs/                # proposal, literature review, progress reports
├── requirements.txt
└── README.md
```

## Data access

SWaT, WADI, and HAI require a signed access request:
- **SWaT / WADI**: request access via iTrust (SUTD) — https://itrust.sutd.edu.sg/itrust-labs_datasets/
- **HAI**: available from the HAI testbed maintainers' repo — https://github.com/icsdataset/hai

## Setup

**For Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**For Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Status

Design phase — see [`docs/progress.md`](docs/progress.md) for current status and deadlines.
