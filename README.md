# Real-Time AI Fraud Decision Engine

Local fraud detection project. Goal is to train a model on transaction data, serve it through a
FastAPI endpoint, and turn the model's fraud probability into a 3-way decision: approve,
review, or block.

Status: Day 1, just getting the scaffold and docs in place. Most sections below are still
placeholders - I'll fill them in as the actual work gets done.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the design and [`tasks.md`](./tasks.md) for
the day-by-day plan.

---

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [API](#api)
- [Monitoring](#monitoring)
- [Local Setup](#local-setup)
- [Next Steps](#next-steps)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

The idea: build a small end-to-end fraud scoring system, not just a model in a notebook.
Data pipeline, trained model, an API that serves it, some experiment tracking, a Dockerfile,
and basic logging/drift checks. Nothing fancy - the point is to have a real working system I
can point to and explain, not a from-scratch research project.

Rough scope:
- Train a classifier (starting with logistic regression, then XGBoost) on a public
  transaction dataset.
- Serve predictions through `POST /score_transaction` in FastAPI. Aiming for well under
  50ms per request, but I haven't measured this yet.
- Turn the raw fraud probability into approve / review / block using two thresholds.
- Explain predictions with SHAP (planned - not implemented yet).
- Track training runs in MLflow.
- Run the API in Docker.
- Log predictions and do a basic drift check against the training data.

Most of this is still TODO - see [`tasks.md`](./tasks.md) for what's actually done vs.
planned.

---

## Business Problem

The reason for a 3-way decision instead of plain fraud/not-fraud: a single threshold forces
every transaction into one of two buckets, but a lot of transactions aren't confidently
either. Blocking a real customer costs you a customer. Letting real fraud through costs you
money directly. Treating everything the same way with one cutoff ignores that.

So instead of one threshold, there are two:

| Decision | Meaning | What happens next |
|---|---|---|
| Approve | Low risk score | Goes through automatically |
| Review | Score in the middle | Gets flagged for manual review / extra auth |
| Block | High risk score | Auto-declined |

Where exactly `T1` and `T2` land isn't decided yet - that happens after I have real
precision/recall numbers to look at (Day 3). Right now these are just placeholders in the
design.

---

## Architecture

Full write-up in [`ARCHITECTURE.md`](./ARCHITECTURE.md) - components, training pipeline,
inference flow, monitoring, and what's in scope vs. cut for later.

Rough flow (offline training → online serving):

```
[Raw transaction data] -> [data pipeline] -> [training pipeline] -> [saved model]
                                                                          |
                                                                          v
                                                          [FastAPI /score_transaction]
                                                                          |
                                                                          v
                                                    [risk_score + decision + top_features]
                                                                          |
                                                                          v
                                              [prediction log] -> [drift check]
```

TODO: replace this with an actual diagram image once the system is built (Day 6).

---

## Dataset

- Source: [Kaggle - Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Size: 284807 rows, 31 columns
- Label distribution: 0.1727% fraudulent / 99.8273% non-fraudulent (highly imbalanced)
- Key features: `Time`, `Amount`, `V1`-`V28` (PCA-anonymized features), `Class` (target)
- License / usage notes: Public Kaggle dataset, for research/educational use. Features `V1`-`V28` are already PCA-transformed and anonymized by the dataset provider - no access to the original raw features.
- EDA notebook: [`notebooks/01_eda.ipynb`](./notebooks/01_eda.ipynb)

---

## Model Training

Nothing trained yet. Table below to fill in once I have real numbers from Day 2/3.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | TBD | TBD | TBD | TBD |
| XGBoost (tuned) | TBD | TBD | TBD | TBD |

- Class imbalance handling: TODO (probably class weights first, imbalanced-learn if that's
  not enough)
- Decision thresholds: TODO - will document the reasoning once picked, not just the numbers
- Explainability: SHAP, planned for Day 3
- Experiment tracking: MLflow, planned for Day 5 (runs stored locally in `mlruns/`, not
  committed to git)
- Notebook: [`notebooks/02_modeling.ipynb`](./notebooks/02_modeling.ipynb) (not started)

---

## API

Not built yet (Day 4). Documenting the intended shape now so I don't drift from it later.

Base URL (local): `http://localhost:8000`

### `GET /health`
Just checks the model loaded and the service is up.

### `POST /score_transaction`

Planned request:
```json
{
  "transaction_id": "txn_123",
  "amount": 249.99,
  "time": "2026-07-12T14:32:00Z"
}
```

Planned response:
```json
{
  "transaction_id": "txn_123",
  "risk_score": 0.87,
  "decision": "block",
  "top_features": [
    {"feature": "amount", "contribution": 0.32}
  ],
  "model_version": "v1.0",
  "timestamp": "2026-07-12T14:32:00Z"
}
```

Once it's running:
```bash
uvicorn src.api.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/score_transaction \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_123", "amount": 249.99, "time": "2026-07-12T14:32:00Z"}'
```

---

## Monitoring

Not built yet (Day 5). Plan:

- Every scored transaction gets appended to a log (CSV or SQLite, haven't decided) via
  `src/monitoring/log_predictions.py`
- A separate script/notebook compares recent transaction features against the training
  distribution - either a hand-rolled check or Evidently, depending on time
- Output is a basic report/plot saved locally, screenshotted for the README later

---

## Local Setup

### Requirements
- Python 3.11+
- Docker (optional, only needed if you want the containerized version)

### Python
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```
(API doesn't exist yet as of Day 1 - this is the intended command once it does.)

### Docker
```bash
docker build -t fraud-engine -f docker/Dockerfile .
docker run -p 8000:8000 fraud-engine
```
(Dockerfile not written yet - planned for Day 5.)

### Tests
```bash
pytest tests/
```
(No tests written yet.)

### MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns
```
Then open `http://localhost:5000`. (Nothing tracked yet.)

---

## Next Steps

Full task breakdown in [`tasks.md`](./tasks.md). Where things stand:

- [x] Day 1: repo scaffold, requirements.txt, README/ARCHITECTURE/tasks docs
- [ ] Day 1: pick dataset, download it, basic EDA
- [ ] Day 2: preprocessing pipeline + baseline model
- [ ] Day 3: tuned model, thresholds, SHAP
- [ ] Day 4: FastAPI service
- [ ] Day 5: MLflow, Docker, monitoring
- [ ] Day 6: docs, demo, cleanup

---

## Limitations & Future Work

Writing this now so I remember to be upfront about it later, not just at the end:

- This uses a static dataset, not an actual real-time transaction stream. "Real-time" here
  means the API responds fast, not that it's hooked up to live data.
- Drift monitoring is a manual/periodic check, not a live dashboard or alerting.
- Thresholds are tuned on one dataset snapshot - would need to be re-checked against live
  data if this were ever actually deployed.
- No auth or rate limiting on the API. Fine for a local demo, not fine for production.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md#8-v1-scope-vs-stretch-scope) for what's in scope
for this project vs. what I'm deliberately skipping.

---

## Tech Stack

Python 3.11, pandas, scikit-learn, XGBoost, imbalanced-learn, SHAP, FastAPI, Pydantic,
MLflow, Docker, Evidently, pytest.

Why each one is here: see [`ARCHITECTURE.md`](./ARCHITECTURE.md#10-tech-stack-rationale).