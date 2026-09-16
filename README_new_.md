# 🛡️ ClaimSense — Insurance Claim Fraud & Severity Intelligence

End-to-end machine learning system for motor-insurance claims. For every incoming claim it answers three questions a claims manager actually asks:

1. **How likely is this claim to be fraudulent?** (classification, severe class imbalance)
2. **How much is it likely to cost us?** (regression on a right-skewed target)
3. **Why did the model say that?** (per-claim explanation, because "the model said so" does not survive an audit)

The models are served behind a FastAPI service and a Streamlit dashboard, containerised with Docker, and tested in CI.

---

## Why this project exists

Most fraud-detection notebooks stop at "ROC-AUC = 0.87 🎉". That number is not a decision. This repo goes the extra mile that production work demands:

- The operating threshold is chosen by **expected rupees saved**, not by accuracy or a default `0.5`.
- Every feature that touches the target is **explicitly dropped for the severity model** — leakage is tested, not assumed.
- Predictions come with **signed per-claim drivers**, plus a **fairness slice check** across gender and state.
- A **drift monitor** tells you when the model has gone stale.

---

## Results

Trained on 20,000 claims with an 11.2% fraud rate, 80/20 stratified split.

### Fraud detection

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Majority-class baseline | 0.112 | 0.500 |
| Logistic regression (balanced) | 0.306 | 0.770 |
| **Gradient-boosted trees** | **0.453** | **0.827** |

5-fold CV PR-AUC: **0.402 ± 0.010** (stable, not a lucky split).

Two operating points, same model:

| Objective | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Maximise F1 | 0.61 | 0.432 | 0.502 | 0.464 |
| **Maximise net savings** | **0.15** | 0.157 | **0.967** | 0.270 |

The cost-optimal threshold deliberately accepts low precision. With an investigation costing ₹250 and an average fraud loss avoided of ₹6,500, a missed fraud is **26× more expensive** than a wasted review — so the business wants recall. Modelled net saving on the 4,000-claim test set: **₹21.2 lakh**.

### Claim severity

| Model | MAE | RMSLE | R² | MAPE |
|---|---|---|---|---|
| Ridge regression (log target) | ₹17,638 | 0.396 | 0.616 | 32.8% |
| **Gradient-boosted trees** | **₹11,819** | **0.302** | **0.858** | **24.7%** |

The tree model wins because repair cost depends on severity *and* vehicle age *together* — an interaction a linear model cannot see.

### Fairness check

Flag rate 68.9% (F) vs 69.1% (M); recall 0.977 (F) vs 0.960 (M). No meaningful disparity at the chosen threshold.

> All numbers are reproducible: `make data && make train` regenerates `reports/metrics.json`.

---

## Architecture

```
raw claims ─▶ validation ─▶ feature engineering ─▶ ┬─ fraud classifier ──┐
                                                   └─ severity regressor ─┤
                                                                          ▼
                                            explainability + threshold policy
                                                                          │
                                        ┌─────────────────────────────────┴───┐
                                        ▼                                     ▼
                                  FastAPI service                    Streamlit dashboard
                                        │                                     │
                                        └──────────── drift monitor ──────────┘
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Data | pandas, NumPy, custom schema validation |
| Modelling | scikit-learn pipelines, LightGBM / XGBoost (auto-detected, graceful fallback to `HistGradientBoosting`) |
| Evaluation | PR-AUC, ROC-AUC, Brier score, MAE/RMSLE/MAPE, cost-based threshold sweep, stratified CV |
| Explainability | SHAP with a permutation/occlusion fallback |
| Monitoring | PSI + total-variation drift, optional Evidently HTML report |
| Serving | FastAPI + Pydantic v2, Streamlit |
| Ops | Docker, docker-compose, GitHub Actions, pytest, ruff, pre-commit, MLflow (optional) |

---

## Quickstart

```bash
git clone https://github.com/<your-username>/claimsense.git
cd claimsense

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m claimsense.data.make_dataset --rows 20000  # generate the dataset
python -m claimsense.models.train                    # train both models
pytest -q                                            # 21 tests
```

Then pick a surface:

```bash
uvicorn claimsense.api.main:app --reload    # API  → http://localhost:8000/docs
streamlit run app/streamlit_app.py          # UI   → http://localhost:8501
python -m claimsense.monitoring             # drift report
docker compose up --build                   # both, containerised
```

### Scoring a claim

```bash
curl -X POST http://localhost:8000/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"CLM100001","customer_age":24,"policy_tenure_months":3,
       "annual_premium":18500,"num_prior_claims":3,"vehicle_age":2,
       "incident_type":"Theft","incident_severity":"Total Loss","incident_hour":2,
       "witnesses":0,"police_report_filed":0,"days_policy_to_incident":61,
       "claim_amount":265000}'
```

```json
{
  "claim_id": "CLM100001",
  "fraud_probability": 0.9674,
  "flagged": true,
  "risk_band": "High",
  "threshold": 0.15,
  "top_drivers": [
    {"feature": "claim_to_premium_ratio", "contribution": 0.059, "direction": "increases risk"},
    {"feature": "days_policy_to_incident", "contribution": 0.020, "direction": "increases risk"},
    {"feature": "prior_claim_rate", "contribution": 0.015, "direction": "increases risk"}
  ]
}
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + whether artefacts are loaded |
| `POST /predict/fraud` | fraud probability, risk band, top drivers |
| `POST /predict/severity` | expected payout |
| `POST /score` | batch scoring + routing decision (STP / desk review / SIU) |

---

## Repository layout

```
claimsense/
├── src/claimsense/
│   ├── config.py                  # paths, column groups, cost assumptions
│   ├── data/make_dataset.py       # synthetic claims generator
│   ├── data/validate.py           # schema + range + duplicate checks
│   ├── features/build_features.py # domain features, preprocessing pipeline
│   ├── models/estimators.py       # model factory with backend fallback
│   ├── models/train.py            # training entrypoint
│   ├── models/evaluate.py         # metrics, threshold sweep, fairness slices
│   ├── models/predict.py          # inference helpers
│   ├── explain/explainer.py       # SHAP / permutation explanations
│   ├── monitoring.py              # PSI + TVD drift detection
│   └── api/                       # FastAPI app + Pydantic schemas
├── app/streamlit_app.py           # 4-tab dashboard
├── notebooks/01_exploratory_analysis.ipynb
├── tests/                         # 21 tests: data, features, metrics, drift, API
├── docs/
│   ├── FEATURE_DICTIONARY.md  # every column, and the leakage rules
│   ├── GITHUB_SETUP.md        # push checklist
│   └── INTERVIEW_NOTES.md     # how to talk about this project
├── .github/workflows/ci.yml       # lint → train → test on 3.10 & 3.11
├── Dockerfile · docker-compose.yml · Makefile
```

---

## The data

The repo ships a **synthetic generator** so the pipeline runs with zero downloads and no licensing headaches. The generating process encodes realistic fraud signals (short tenure, prior-claim history, night incidents, no police report, few witnesses) plus messy missing values.

To run on real data instead, point `load_raw()` at a Kaggle dataset — [Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) or [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity) — and map the columns in `config.py`. Nothing else changes.

---

## Engineered features

| Feature | Definition | Why it matters |
|---|---|---|
| `claim_to_premium_ratio` | claim ÷ annual premium | exposure-normalised claim size; a small policy with a huge claim is the classic pattern |
| `is_night_incident` | incident between 23:00–05:00 | fewer witnesses, harder to verify |
| `prior_claim_rate` | prior claims ÷ policy years | frequency beats raw count for new policies |
| `days_policy_to_incident` | days between inception and loss | very early losses are a known red flag |
| `is_weekend_incident` | Sat/Sun flag | different accident mix |
| `young_driver_flag` | age < 25 | different risk profile |

Full definitions in [`docs/FEATURE_DICTIONARY.md`](docs/FEATURE_DICTIONARY.md).

---

## Design decisions worth defending in an interview

- **Accuracy is banned.** At an 11% base rate, predicting "never fraud" scores 89%. Everything is tuned on PR-AUC.
- **Leakage is structural, not incidental.** `feature_frame()` drops `claim_to_premium_ratio` for the severity task because it is derived from the target — and `test_features.py` fails the build if that ever regresses.
- **Log-transformed target.** Claim amounts are heavily right-skewed; training on `log1p` and inverting at predict time cut MAE by roughly a third.
- **Threshold is a business decision.** The cost matrix lives in `config.py`, so an actuary can change the assumption without touching the model.
- **Imbalance handled by weighting, not blind SMOTE.** `scale_pos_weight` / `class_weight="balanced"` keeps the probability calibration interpretable (Brier score is reported).
- **Backend fallback.** The model factory uses LightGBM if it is available, XGBoost if not, and scikit-learn otherwise — so a reviewer can clone and run without a build toolchain.

---

## Limitations & next steps

- Synthetic data means the absolute metrics are optimistic; the pipeline, not the score, is the deliverable.
- No temporal validation split — real claims data would need a time-based split to avoid look-ahead bias.
- Next: Optuna tuning, probability calibration (isotonic), a claim-document NLP feature from adjuster notes, and a scheduled retraining job triggered by the drift monitor.

---

## License

MIT — see [LICENSE](LICENSE).
