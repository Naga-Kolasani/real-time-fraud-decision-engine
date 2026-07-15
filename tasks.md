# 6-Day Build Plan

🔴 = need this, 🟡 = if there's time,
`needs:` = has to happen after something else.

---

## Day 1 - Setup, dataset, architecture

**Env**
- [ ] 🔴 GitHub repo created, cloned locally
- [ ] 🔴 venv set up, Python 3.11+
- [ ] 🔴 requirements.txt installed, imports verified

**Scaffold**
- [ ] 🔴 Folder structure (`notebooks/`, `src/{data,models,api,monitoring}/`, `tests/`,
  `docker/`, `artifacts/`) - `needs:` env
- [ ] 🔴 `__init__.py` files where needed
- [ ] 🔴 `.gitignore`

**Dataset**
- [ ] 🔴 Pick a dataset (leaning Kaggle credit card fraud, might switch to IEEE-CIS)
- [ ] 🔴 Download it, keep it out of git
- [ ] 🔴 `01_eda.ipynb`: shape, columns, label balance, missing values - `needs:` dataset
- [ ] 🔴 Note dataset source in README

**Docs**
- [ ] 🔴 `ARCHITECTURE.md` - problem framing, flow, success criteria
- [ ] 🔴 Success criteria written down (target PR-AUC range, <50ms latency budget)
- [ ] 🔴 This file

**Done when:** repo pushed, env works, dataset loaded with basic stats, ARCHITECTURE.md and
tasks.md exist.

---

## Day 2 - Data prep, baseline model

**EDA**
- [ ] 🔴 Feature distributions in `01_eda.ipynb` - `needs:` Day 1 dataset
- [ ] 🔴 Missing values / data quality check
- [ ] 🔴 Class imbalance visualized
- [ ] 🔴 Decide train/val/test split (stratified)

**Pipeline**
- [ ] 🔴 `src/data/load_data.py` - `needs:` dataset picked
- [ ] 🔴 `src/data/preprocess.py` - scaling/encoding
- [ ] 🔴 Wrap into one sklearn `Pipeline` object - `needs:` preprocess.py

**Baseline**
- [ ] 🔴 Logistic regression baseline in `02_modeling.ipynb` - `needs:` pipeline
- [ ] 🔴 XGBoost as the "real" model
- [ ] 🔴 Handle imbalance (class weights, or imbalanced-learn if needed)
- [ ] 🔴 Evaluate: precision, recall, F1, PR-AUC, confusion matrix
- [ ] 🔴 Save metrics + model artifact to `artifacts/models/`

**Done when:** preprocess.py works, baseline + XGBoost trained, notebook has metrics/plots,
first model saved.

---

## Day 3 - Tuning, thresholds, explainability

**Tuning**
- [ ] 🔴 Quick hyperparameter search on XGBoost (not exhaustive) - `needs:` Day 2 model
- [ ] 🔴 Pick final model, balancing recall vs. false positives

**Thresholds**
- [ ] 🔴 `src/models/infer.py` - risk_score from predict_proba - `needs:` final model
- [ ] 🔴 T1/T2 threshold logic
- [ ] 🔴 Precision/recall vs. threshold plot to back up where T1/T2 landed

**Explainability**
- [ ] 🔴 SHAP global importance in `02_modeling.ipynb` - `needs:` final model
- [ ] 🔴 Local explanations for a few sample transactions
- [ ] 🔴 Top-N feature function in `infer.py`
- [ ] 🟡 Check if SHAP is fast enough for the latency budget, or if it needs to be
  approximated

**Done when:** tuned model saved, thresholds picked and written down with reasoning,
explainability working in the notebook, `infer.py` has the core functions.

---

## Day 4 - API

**Schema**
- [ ] 🔴 `src/api/schemas.py` - request/response fields - `needs:` Day 3 infer.py

**FastAPI**
- [ ] 🔴 `src/api/main.py` - load model at startup - `needs:` Day 2-3 artifacts
- [ ] 🔴 `GET /health`
- [ ] 🔴 `POST /score_transaction`
- [ ] 🔴 Test manually with curl/Postman
- [ ] 🔴 `tests/test_inference.py`
- [ ] 🟡 `tests/test_data_pipeline.py`

**Extra**
- [ ] 🟡 Small script to batch-score a CSV - `needs:` working infer logic
- [ ] 🟡 Save sample outputs for the README

**Done when:** API runs locally, both endpoints work, manual tests documented.

---

## Day 5 - MLflow, Docker, monitoring

**MLflow**
- [ ] 🔴 Wire into `train.py` - log params/metrics/artifacts - `needs:` Day 2-3 training code
- [ ] 🔴 Save model via MLflow logging
- [ ] 🟡 Simple "production model" tag/config convention

**Docker**
- [ ] 🔴 `docker/Dockerfile` - `needs:` Day 4 working API
- [ ] 🔴 Build + run locally
- [ ] 🔴 Confirm `/health` and `/score_transaction` work inside the container
- [ ] 🟡 `docker-compose.yml`

**Monitoring**
- [ ] 🔴 `src/monitoring/log_predictions.py` - `needs:` Day 4 API
- [ ] 🔴 `drift_checks.py` or `03_monitoring_and_drift.ipynb` - `needs:` some logged
  predictions to exist
- [ ] 🔴 At least one drift plot/report saved to `artifacts/metrics/`
- [ ] 🟡 Evidently instead of hand-rolled checks, if time allows

**Done when:** MLflow tracking runs, Docker image runs the API, predictions get logged, one
drift check/report exists.

---

## Day 6 - Docs, demo, cleanup

**README**
- [ ] 🔴 Fill in all the TODO sections with real content - `needs:` Days 1-5 done
- [ ] 🔴 Swap the ASCII diagram for a real image
- [ ] 🔴 Real metrics in the model table
- [ ] 🔴 Real curl examples with actual output

**Demo material**
- [ ] 🔴 Screenshot: architecture diagram
- [ ] 🔴 Screenshot: notebook plots (feature importance, confusion matrix, drift report)
- [ ] 🔴 Screenshot: API call + response
- [ ] 🟡 Short screen recording walking through it

**Cleanup**
- [ ] 🔴 Run full pytest suite, fix failures
- [ ] 🔴 Double check no data files or secrets got committed
- [ ] 🔴 Fresh clone, follow my own README from scratch, make sure it actually works -
  `needs:` everything else done
- [ ] 🟡 GitHub Actions running pytest on push

**Wrap-up**
- [ ] 🔴 1-2 resume bullets
- [ ] 🔴 Pin the repo
- [ ] 🟡 Short LinkedIn post

**Done when:** README matches reality, tests pass on a clean checkout, Docker works, repo is
ready to link somewhere.

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

If a day runs over, protect Day 4 (API) and Day 6 (docs/cleanup) first - a working API with
an honest README beats a slightly better model with no way to show it off.

---

## Must-have list for Saturday

- [ ] Trained model with real metrics written down
- [ ] Working `/score_transaction` endpoint returning approve/review/block
- [ ] README that matches what's actually built
- [ ] Dockerfile that runs the API with one command
- [ ] Prediction logging + at least one drift check
- [ ] MLflow tracking on the training runs
- [ ] pytest passes
- [ ] Resume bullets + pinned repo