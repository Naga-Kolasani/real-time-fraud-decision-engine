# Architecture - Real-Time AI Fraud Decision Engine

Architecture notes for the implemented local fraud-scoring system. This document describes the current v1 design, its local deployment workflow, and the deliberate limits of the project.

## 1. Goal

Build a working fraud-scoring system end to end: data in, model trained, model served over an API, decision returned as approve/review/block, with basic tracking and monitoring around it. Local only, no paid services. Six days, working solo.

This is not meant to be a novel modeling approach. The goal is a complete, working local system with the core pieces of a small deployment: a pipeline, service, tracking, container, and monitoring.

## 2. Business Problem

A plain fraud/not-fraud classifier forces every transaction through one threshold. In practice, many transactions are ambiguous and the cost of getting it wrong is not symmetric. Missing fraud costs money directly, while blocking a legitimate customer can cost you that customer. So instead of one cutoff, use two:

| Decision | Meaning | Downstream action |
|---|---|---|
| Approve | risk_score < T1 | Auto-process |
| Review | T1 <= risk_score < T2 | Manual review / step-up auth |
| Block | risk_score >= T2 | Auto-decline |

T1 and T2 get picked from actual precision/recall curves on the validation set, not guessed up front. The current working values are `T1 = 0.10` and `T2 = 0.89`, based on the validation sweep plus a manual adjustment to keep the review band from swallowing almost the entire validation set.

## 3. End-to-End Flow

Two separate paths: an offline one (training, run manually/occasionally) and an online one (the API, running continuously).

```
                     ┌──────────────────────────┐
                     │   Raw Transaction Data   │
                     │  (public fraud dataset)  │
                     └───────────┬──────────────┘
                                 │
                                 v
                    ┌───────────────────────────┐
                    │  Data pipeline (offline)  │
                    │  load -> clean -> split   │
                    └────────────┬──────────────┘
                                 │
                                 v
                   ┌─────────────────────────────┐
                   │      Training pipeline      │
                   │   preprocess -> train ->    │
                   │  evaluate -> threshold ->   │
                   │   log to MLflow -> save     │
                   └─────────────┬───────────────┘
                                 │
                                 v
                ┌────────────────────────────────────┐
                │   Saved model + pipeline object    │
                │   (artifacts/models/*.pkl)         │
                └────────────────┬───────────────────┘
                                 │
                                 v
   ┌──────────────────────────────────────────────────────────────┐
   │                 FastAPI service (online)                     │
   │  POST /score_transaction                                     │
   │  JSON -> feature vector -> model.predict_proba()             │
   │  -> risk_score -> threshold decision -> response metadata    │
   └─────────────────────────────┬────────────────────────────────┘
                                 │
                                 v
                    ┌────────────────────────────┐
                    │   Prediction log (JSONL)   │
                    └────────────┬───────────────┘
                                 │
                                 v
                    ┌────────────────────────────┐
                    │   Manual PSI drift check   │
                    │    (local JSON report)     │
                    └────────────────────────────┘
```

The offline path trains and tracks candidate models; the online path loads the selected artifact once at startup, scores validated requests, writes successful prediction records, and supports on-demand drift checks.

## 4. Components

| Piece | Where | What it does |
|---|---|---|
| Data loader | `src/data/load_data.py` | Reads raw dataset into a DataFrame |
| Preprocessor | `src/data/preprocess.py` | Cleaning/encoding/scaling, wrapped in a single sklearn `Pipeline` |
| Trainer | `src/models/train.py` | Fits the model(s), logs to MLflow, saves the artifact |
| Inference logic | `src/models/infer.py` | Loads the saved artifact, returns a fraud risk score, and applies approve/review/block threshold logic |
| API | `src/api/main.py`, `src/api/schemas.py` | FastAPI app - `/health`, `/score_transaction` |
| Prediction logger | `src/monitoring/log_predictions.py` | Appends each successful scored request and its output to `artifacts/logs/predictions.jsonl` |
| Drift checker | `src/monitoring/drift_checks.py` | Computes PSI for logged `Amount`, `V1`, `V2`, `V3`, and `V4` values against the training reference and writes `artifacts/metrics/drift_report.json` |
| Notebooks | `notebooks/01_eda.ipynb`, `notebooks/02_modeling.ipynb` | Exploratory EDA plus model evaluation and SHAP explainability; not imported by the API |

Reasoning for splitting notebooks vs. `src/`: anything that needs to run more than once (preprocessing, training, inference, monitoring) goes in `src/` as a plain importable module so it can be tested and reused by the API. Notebooks are just for looking at things.

## 5. Training Pipeline

1. Load raw data.
2. Split into train/val/test - stratified, since the classes are imbalanced.
3. Preprocess (scale/encode), fit only on train, wrapped in one sklearn `Pipeline` so the exact same transform gets reused at inference time. This is the main thing that prevents train/serve mismatch later.
4. Handle class imbalance with `class_weight="balanced"` for Logistic Regression and training-split-derived `scale_pos_weight` for XGBoost.
5. Train: logistic regression baseline first (cheap sanity check), then XGBoost.
6. Evaluate on the held-out test set. Using PR-AUC as the main number since ROC-AUC looks good even for bad models when classes are this imbalanced.
7. Pick T1/T2 by looking at precision/recall vs. threshold plots and write down why, not just the numbers. Right now the working cutoffs are `T1 = 0.10` and `T2 = 0.89`; the raw sweep suggested `T1 = 0.00`, but that made the review band too large to be useful.
8. Use SHAP in the modeling notebook for global importance and local explanations on sample transactions.
9. Log each model's parameters, metrics, and confusion matrix to local MLflow; log the winning model artifact in its nested run and save the deployable artifact to `artifacts/models/model_v1.pkl`.

## 6. Inference / API Flow

`POST /score_transaction`:

1. Validates the 30 numeric inputs (`Time`, `Amount`, and `V1` through `V28`) with the Pydantic request schema. Non-finite values and missing fields are rejected before scoring.
2. Loads the saved model artifact once when FastAPI starts. The artifact contains both the fitted preprocessing pipeline and selected model.
3. Applies the saved preprocessing pipeline and gets `P(fraud)` from `predict_proba()` as `risk_score`.
4. Applies the current threshold policy:
   - `risk_score < T1` → `approve`
   - `T1 <= risk_score < T2` → `review`
   - `risk_score >= T2` → `block`
   - Current values are `T1 = 0.10` and `T2 = 0.89`.
5. Returns `risk_score`, `decision`, `model_name`, `model_version`, and `timestamp`.
6. Appends successful scoring events to the local JSONL prediction log.

SHAP explanations exist in the modeling notebook but are not calculated or returned on the API path. This keeps the v1 response focused on scoring and threshold decisions.

Latency is a target rather than a measured claim: the goal is under 50ms for scoring after the one-time model load.

`GET /health` confirms the model artifact loaded and returns the model name and version.

## 7. Monitoring / Logging

Successful scoring requests append one JSON Lines record to:

```text
artifacts/logs/predictions.jsonl
```

Each record includes the validated transaction fields, `risk_score`, `decision`, model metadata, and timestamps. Requests rejected by request validation are not logged.

The Docker Compose configuration bind-mounts the host `artifacts/logs/` directory to `/app/artifacts/logs`, so prediction logs persist after the container is removed.

Run the local drift check on demand:

```bash
python -m src.monitoring.drift_checks
```

The script loads the training dataset as the reference distribution and compares logged values for `Amount`, `V1`, `V2`, `V3`, and `V4` with Population Stability Index (PSI). A feature is flagged when PSI is at least `0.20`. The script prints a summary and writes:

```text
artifacts/metrics/drift_report.json
```

This is intentionally lightweight local monitoring: no live dashboard, alerting, or claim that a tiny local demonstration is evidence of real production drift.

## 8. V1 Scope vs. Stretch Scope

### V1 - needs to exist by Day 6
- Cleaned dataset + EDA
- Preprocessing pipeline (sklearn `Pipeline`)
- Baseline + selected model, with real metrics written down
- Threshold logic with documented reasoning
- SHAP explainability in the modeling notebook (global + sample local explanations)
- FastAPI: `/health` + `/score_transaction`
- MLflow tracking
- Dockerfile that runs the API
- Prediction logging
- At least one drift check/report
- README that actually reflects what's built
- 1-2 resume bullets

### Stretch - only if there's time left
- Local Kafka-like streaming simulator
- Real Evidently dashboard (vs. a static report)
- Small frontend (Streamlit or plain HTML) hitting the API
- MLflow model registry / "production" tag convention
- GitHub Actions running pytest on push
- Comparing two model versions side by side

## 9. Success Criteria

Model:
- PR-AUC clearly better than the naive baseline (just guessing the majority class)
- Recall on the fraud class matters more than raw accuracy, given the imbalance
- Thresholds documented with reasoning, not just picked arbitrarily

System:
- Target: keep `/score_transaction` under 50ms locally after model load; this has not been benchmarked yet
- Same API behavior running via `uvicorn` directly and via Docker Compose
- Every prediction gets logged; drift check can be run on demand against that log
- `pytest` passes locally; a fresh-clone verification is still a Day 6 task

Docs/repo:
- README is understandable without needing to read the code first
- No large data files or secrets committed
- Someone could clone this and get it running from the README alone

## 10. Tech Stack Rationale

| Choice | Why |
|---|---|
| sklearn Pipeline | Keeps preprocessing identical between training and inference, which avoids a common bug where the two drift apart |
| XGBoost | Works well on tabular data, trains fast locally, common choice for this kind of problem |
| imbalanced-learn | The dataset will be imbalanced; worth handling explicitly instead of ignoring it |
| SHAP | Standard tool for explaining individual predictions, relevant since fraud decisions usually need some justification |
| FastAPI + Pydantic | Async support, built-in request validation, auto docs; reasonable default for serving a model over HTTP |
| MLflow | Free, runs locally, keeps track of runs/params/metrics without needing any cloud account |
| Docker | Makes sure "works on my machine" isn't a problem for anyone else running it |
| Custom PSI drift check | Keeps monitoring local and dependency-light while comparing selected logged features with the training reference distribution; Evidently remains possible for future work |
| No paid APIs, no required cloud | Keeps it reproducible by anyone who clones the repo, no signup or billing needed |