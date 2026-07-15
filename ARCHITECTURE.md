# Architecture - Real-Time AI Fraud Decision Engine

Notes on how this is put together, written before most of it exists. Treat this as the plan
I'm building against, not a description of a finished system - I'll update it if reality
diverges (it probably will, a bit).

## 1. Goal

Build a working fraud-scoring system end to end: data in, model trained, model served over
an API, decision returned as approve/review/block, with basic tracking and monitoring around
it. Local only, no paid services. Six days, working solo.

This isn't trying to be a novel modeling approach - it's trying to be a complete, working
system with the pieces you'd expect in a real (small) deployment: a pipeline, a service, some
tracking, a container, some logging.

## 2. Business Problem

A plain fraud/not-fraud classifier forces every transaction through one threshold. In
practice a lot of transactions are ambiguous, and the cost of getting it wrong isn't
symmetric - missing fraud costs money directly, blocking a real customer costs you the
customer. So instead of one cutoff, use two:

| Decision | Meaning | Downstream action |
|---|---|---|
| Approve | risk_score < T1 | Auto-process |
| Review | T1 <= risk_score < T2 | Manual review / step-up auth |
| Block | risk_score >= T2 | Auto-decline |

T1 and T2 get picked from actual precision/recall curves on the validation set (Day 3), not
guessed up front.

## 3. End-to-End Flow

Two separate paths: an offline one (training, run manually/occasionally) and an online one
(the API, running continuously).

```
                     ┌────────────────────────┐
                     │   Raw Transaction Data   │
                     │  (public fraud dataset)  │
                     └───────────┬──────────────┘
                                 │
                                 v
                   ┌─────────────────────────┐
                   │  Data pipeline (offline)  │
                   │  load -> clean -> split   │
                   └───────────┬───────────────┘
                                 │
                                 v
                   ┌─────────────────────────┐
                   │   Training pipeline       │
                   │ preprocess -> train ->    │
                   │ evaluate -> threshold ->  │
                   │ log to MLflow -> save     │
                   └───────────┬───────────────┘
                                 │
                                 v
                 ┌───────────────────────────────┐
                 │   Saved model + pipeline object │
                 │   (artifacts/models/*.pkl)      │
                 └───────────────┬─────────────────┘
                                 │
                                 v
   ┌───────────────────────────────────────────────────────┐
   │                 FastAPI service (online)                 │
   │  POST /score_transaction                                  │
   │  JSON -> feature vector -> model.predict_proba()           │
   │  -> risk_score -> threshold logic -> top_features (SHAP)   │
   └───────────────────────┬───────────────────────────────┘
                                 │
                                 v
                 ┌───────────────────────────────┐
                 │  Prediction log (CSV/SQLite)     │
                 └───────────────┬─────────────────┘
                                 │
                                 v
                 ┌───────────────────────────────┐
                 │  Drift check (Evidently or        │
                 │  hand-rolled comparison)          │
                 └───────────────────────────────┘
```

Days 2–3 build the top half (offline). Days 4–5 build the bottom half (online).

## 4. Components

| Piece | Where | What it does |
|---|---|---|
| Data loader | `src/data/load_data.py` | Reads raw dataset into a DataFrame |
| Preprocessor | `src/data/preprocess.py` | Cleaning/encoding/scaling, wrapped in a single sklearn `Pipeline` |
| Trainer | `src/models/train.py` | Fits the model(s), logs to MLflow, saves the artifact |
| Evaluator | `src/models/evaluate.py` | Precision/recall/F1/PR-AUC, confusion matrix |
| Inference logic | `src/models/infer.py` | risk_score, threshold decision, top-feature explanation |
| API | `src/api/main.py`, `src/api/schemas.py` | FastAPI app - `/health`, `/score_transaction` |
| Prediction logger | `src/monitoring/log_predictions.py` | Appends each scored request to a log file |
| Drift checker | `src/monitoring/drift_checks.py` | Compares recent vs. training feature distributions |
| Notebooks | `notebooks/01-03` | EDA, modeling/explainability, monitoring - exploratory, not imported by anything else |

Reasoning for splitting notebooks vs. `src/`: anything that needs to run more than once
(preprocessing, training, inference, monitoring) goes in `src/` as a plain importable module
so it can be tested and reused by the API. Notebooks are just for looking at things.

## 5. Training Pipeline

1. Load raw data.
2. Split into train/val/test - stratified, since the classes are imbalanced.
3. Preprocess (scale/encode), fit only on train, wrapped in one sklearn `Pipeline` so the
   exact same transform gets reused at inference time. This is the main thing that prevents
   train/serve mismatch later.
4. Deal with class imbalance - probably class weights first since it's simpler; will try
   imbalanced-learn resampling if that's not enough.
5. Train: logistic regression baseline first (cheap sanity check), then XGBoost.
6. Evaluate on the held-out test set. Using PR-AUC as the main number since ROC-AUC looks
   good even for bad models when classes are this imbalanced.
7. Pick T1/T2 by looking at precision/recall vs. threshold plots - write down why, not just
   the numbers.
8. SHAP for explainability - global importance plus per-prediction breakdown.
9. Log everything to MLflow, save the best model to `artifacts/models/`.

## 6. Inference / API Flow

`POST /score_transaction`:

1. Validate the request against the Pydantic schema.
2. Run it through the same saved preprocessing pipeline used in training.
3. Get `P(fraud)` from the model → `risk_score`.
4. Apply the threshold logic:
   - `risk_score < T1` → approve
   - `T1 <= risk_score < T2` → review
   - `risk_score >= T2` → block
5. Compute top contributing features (SHAP value, or a cheaper approximation if SHAP is too
   slow for the latency budget - need to check this once it's built).
6. Return `transaction_id`, `risk_score`, `decision`, `top_features`, `model_version`,
   `timestamp`.

Latency target: under 50ms for the scoring logic itself (not counting model load at startup,
which only happens once). Haven't measured this yet - it's a target, not a confirmed number.

`GET /health` - just confirms the model's loaded and the process is up.

## 7. Monitoring / Logging

1. Every request to `/score_transaction` gets appended to a local log (CSV or SQLite,
   deciding later) - input features, risk_score, decision, model_version, timestamp.
2. A separate script/notebook periodically compares:
   - Feature distributions of recently logged transactions vs. the training set
     (PSI/KS-test, or just use Evidently if there's time)
   - Whether the block rate is drifting over time
3. Output is a plot or an Evidently HTML report, saved to `artifacts/metrics/` - this is what
   gets screenshotted for the README later.

Deliberately lightweight - no live dashboard, no alerting. Just enough to show the concept
works.

## 8. V1 Scope vs. Stretch Scope

### V1 - needs to exist by Saturday
- Cleaned dataset + EDA
- Preprocessing pipeline (sklearn `Pipeline`)
- Baseline + tuned model, with real metrics written down
- Threshold logic with documented reasoning
- SHAP explainability (global + per-prediction)
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

If a stretch item is putting Saturday at risk, cut it. A finished, working V1 beats a
half-finished system with extra features.

## 9. Success Criteria

Model:
- PR-AUC clearly better than the naive baseline (just guessing the majority class)
- Recall on the fraud class matters more than raw accuracy, given the imbalance
- Thresholds documented with reasoning, not just picked arbitrarily

System:
- `/score_transaction` responds in under 50ms locally (excluding model load)
- Same behavior running via `uvicorn` directly and via `docker run`
- Every prediction gets logged; drift check can be run on demand against that log
- `pytest` passes on a clean checkout

Docs/repo:
- README is understandable without needing to read the code first
- No large data files or secrets committed
- Someone could clone this and get it running from the README alone

## 10. Tech Stack Rationale

| Choice | Why |
|---|---|
| sklearn Pipeline | Keeps preprocessing identical between training and inference - avoids a common bug where the two drift apart |
| XGBoost | Works well on tabular data, trains fast locally, common choice for this kind of problem |
| imbalanced-learn | The dataset will be imbalanced; worth handling explicitly instead of ignoring it |
| SHAP | Standard tool for explaining individual predictions, relevant since fraud decisions usually need some justification |
| FastAPI + Pydantic | Async support, built-in request validation, auto docs - reasonable default for serving a model over HTTP |
| MLflow | Free, runs locally, keeps track of runs/params/metrics without needing any cloud account |
| Docker | Makes sure "works on my machine" isn't a problem for anyone else running it |
| Evidently (maybe) | Purpose-built for this kind of drift check instead of hand-rolling stats from scratch - using it if there's time, otherwise a simpler custom check |
| No paid APIs, no required cloud | Keeps it reproducible by anyone who clones the repo, no signup or billing needed |