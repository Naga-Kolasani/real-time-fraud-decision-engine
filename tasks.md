# 6-Day Build Plan

🔴 = need this, 🟡 = if there's time, `needs:` = has to happen after something else.

---

## Day 1 - Setup, dataset, architecture

**Env**
- [x] 🔴 GitHub repo created, cloned locally
- [x] 🔴 venv set up, Python 3.11+
- [x] 🔴 requirements.txt installed, imports verified

**Scaffold**
- [x] 🔴 Folder structure (`notebooks/`, `src/{data,models,api,monitoring}/`, `tests/`,
  `docker/`, `artifacts/`) - `needs:` env
- [x] 🔴 `__init__.py` files where needed
- [x] 🔴 `.gitignore`

**Dataset**
- [x] 🔴 Pick a dataset (leaning Kaggle credit card fraud, might switch to IEEE-CIS)
- [x] 🔴 Download it, keep it out of git
- [x] 🔴 `01_eda.ipynb`: shape, columns, label balance, missing values - `needs:` dataset
- [x] 🔴 Note dataset source in README

**Docs**
- [x] 🔴 `ARCHITECTURE.md` - problem framing, flow, success criteria
- [x] 🔴 Success criteria written down (target PR-AUC range, <50ms latency budget)
- [x] 🔴 This file

**Done when:** repo pushed, env works, dataset loaded with basic stats, ARCHITECTURE.md and tasks.md exist.

---

## Day 2 - Data prep, baseline model

**EDA**
- [x] 🔴 Feature distributions in `01_eda.ipynb` - `needs:` Day 1 dataset
- [x] 🔴 Missing values / data quality check
- [x] 🔴 Class imbalance visualized
- [x] 🔴 Decide train/val/test split (stratified)

**Pipeline**
- [x] 🔴 `src/data/load_data.py` - `needs:` dataset picked
- [x] 🔴 `src/data/preprocess.py` - scaling/encoding
- [x] 🔴 Wrap into one sklearn `Pipeline` object - `needs:` preprocess.py

**Baseline**
- [x] 🔴 Logistic regression baseline in `02_modeling.ipynb` - `needs:` pipeline
- [x] 🔴 XGBoost as the "real" model
- [x] 🔴 Handle imbalance (class weights, or imbalanced-learn if needed)
- [x] 🔴 Evaluate: precision, recall, F1, PR-AUC, confusion matrix
- [x] 🔴 Save metrics + model artifact to `artifacts/models/`

**Done when:** preprocess.py works, baseline + XGBoost trained, notebook has metrics/plots, first model saved.

---

## Day 3 - Tuning, thresholds, explainability

**Tuning**
- [x] 🔴 Quick hyperparameter search on XGBoost (not exhaustive) - `needs:` Day 2 model
- [x] 🔴 Select XGBoost as the v1 model based on held-out PR-AUC and precision/recall tradeoffs

**Thresholds**
- [x] 🔴 `src/models/infer.py` - risk_score from predict_proba - `needs:` final model
- [x] 🔴 T1/T2 threshold logic
- [x] 🔴 Precision/recall vs. threshold plot to back up where T1/T2 landed

**Explainability**
- [x] 🔴 SHAP global importance in `02_modeling.ipynb` - `needs:` final model
- [x] 🔴 Local explanations for a few sample transactions
- [x] 🔴 Top-N global feature-importance helper in `src/models/infer.py`
- [ ] 🟡 Check if SHAP is fast enough for the latency budget, or if it needs to be approximated

**Done when:** a selected model artifact is saved, working thresholds are documented with reasoning, `infer.py` has the core scoring functions, and explainability is working in the notebook.

---

## Day 4 - API

**Schema**
- [x] 🔴 `src/api/schemas.py` - request/response fields - `needs:` Day 3 infer.py

**FastAPI**
- [x] 🔴 `src/api/main.py` - load model at startup - `needs:` Day 2-3 artifacts
- [x] 🔴 `GET /health`
- [x] 🔴 `POST /score_transaction`
- [x] 🔴 Test manually with curl/Postman
- [x] 🔴 `tests/test_inference.py`
- [x] 🔴 `tests/test_api.py` - health, scoring, and request validation
- [ ] 🟡 `tests/test_data_pipeline.py`

**Extra**
- [ ] 🟡 Small script to batch-score a CSV - `needs:` working infer logic
- [ ] 🟡 Save sample outputs for the README

**Done when:** API runs locally, both endpoints work, manual tests documented.

---

## Day 5 - MLflow, Docker, monitoring

**MLflow**
- [x] 🔴 Wire into `train.py` - log params/metrics/artifacts - `needs:` Day 2-3 training code
- [x] 🔴 Save model via MLflow logging
- [ ] 🟡 Simple "production model" tag/config convention

**Docker**
- [x] 🔴 `docker/Dockerfile` - `needs:` Day 4 working API
- [x] 🔴 Build + run locally
- [x] 🔴 Confirm `/health` and `/score_transaction` work inside the container
- [x] 🟡 `docker-compose.yml`

**Monitoring**
- [x] 🔴 `src/monitoring/log_predictions.py` - `needs:` Day 4 API
- [x] 🔴 `src/monitoring/drift_checks.py` - `needs:` logged predictions
- [x] 🔴 At least one drift plot/report saved to `artifacts/metrics/`
- [ ] 🟡 Evidently instead of hand-rolled checks, if time allows

**Done when:** MLflow tracking runs, the Docker image runs the API, successful predictions are logged, and one drift report can be generated locally.

---

## Day 6 - Docs, demo, cleanup

**README**
- [x] 🔴 Fill in all the TODO sections with real content - `needs:` Days 1-5 done
- [ ] 🔴 Swap the ASCII diagram for a real image
- [x] 🔴 Real metrics in the model table
- [x] 🔴 Real curl examples with actual output

**Demo material**
- [ ] 🔴 Screenshot: architecture diagram
- [x] 🔴 Screenshot: notebook plots (feature importance and threshold sweep)
- [x] 🔴 Screenshot: API call + response
- [ ] 🟡 Short screen recording walking through it

**Cleanup**
- [x] 🔴 Run full pytest suite, fix failures
- [x] 🔴 Double check no data files or secrets got committed
- [x] 🔴 Fresh clone, follow my own README from scratch, make sure it actually works - `needs:` everything else done
- [ ] 🟡 GitHub Actions running pytest on push

**Wrap-up**
- [x] 🔴 1-2 resume bullets
- [x] 🔴 Pin the repo
- [ ] 🟡 Short LinkedIn post

**Done when:** README matches reality, tests pass on a clean checkout, Docker works, repo is ready to link somewhere.

---

## Dependency chain (roughly)

```
Day 1 (env, dataset, docs)
  -> Day 2 (preprocessing, baseline model)
    -> Day 3 (tuning, thresholds, SHAP)
      -> Day 4 (API wraps infer.py)
        -> Day 5 (MLflow wraps training, Docker wraps API, monitoring wraps API logs)
          -> Day 6 (README/demo/tests wrap all of it)
```

---

## Must-have list by Day 6

- [x] Trained model with real metrics written down
- [x] Working `/score_transaction` endpoint returning approve/review/block
- [x] README that matches what's actually built
- [x] Dockerfile that runs the API with one command
- [x] Prediction logging + at least one drift check
- [x] MLflow tracking on the training runs
- [x] pytest passes
- [x] Resume bullets + pinned repo
