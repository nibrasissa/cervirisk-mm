# CerviRisk-MM

**Multi-modal cervical cancer risk prediction — end-to-end pipeline with continuous data ingestion.**

> **Research prototype, not a medical device.** All predictions are tagged `RESEARCH_PROTOTYPE` and accompanied by a clinical disclaimer.

[![Live demo](https://img.shields.io/badge/%F0%9F%8E%97%EF%B8%8F_live_demo-cervirisk--mm.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cervirisk-mm.streamlit.app/)

[![Tests](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/tests.yml/badge.svg)](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/tests.yml)
[![Docker](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/docker-publish.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Try it now

**Live demo:** https://cervirisk-mm.streamlit.app/

Click the link, pick a sample patient profile in the sidebar, click **Run prediction**. Returns a risk probability, tier, and SHAP feature contributions explaining the prediction. No install, no signup, just works.

---

## What this project is

A reproducible machine learning pipeline that:

1. **Ingests** four public data sources on demand (UCI clinical, NCBI HPV sequences, 1000 Genomes ancestry, PGS Catalog)
2. **Assembles** synthetic multi-modal patient records anchored on real UCI outcomes, with explicit provenance tracking
3. **Trains** 18 model variants (3 feature modes × 6 algorithms) under nested LOOCV + 5-fold stability evaluation
4. **Serves** the best variant through a FastAPI service with SHAP feature attribution
5. **Watches** itself with statistical drift detection (PSI / KS / chi-square) against batch and live NCBI feeds
6. **Visualizes** everything through a clinical-style Streamlit decision-support UI

53 automated tests cover every layer. The pipeline is fully reproducible in 5 minutes on a clean machine.

---

## Three ways to use this project

### 1. Live demo — zero setup

Open https://cervirisk-mm.streamlit.app/ and start clicking. The deployed model is loaded; SHAP attribution works; three sample patients are pre-built.

### 2. Docker — full version locally without Python install

```bash
docker pull ghcr.io/nibrasissa/cervirisk-mm:latest
docker run -p 8000:8000 ghcr.io/nibrasissa/cervirisk-mm:latest
# Then open http://localhost:8000/docs for the Swagger UI
```

### 3. Clone — full pipeline including training

```powershell
git clone https://github.com/nibrasissa/cervirisk-mm.git
cd cervirisk-mm
.\run.ps1 quickstart        # install + ingest + assemble + fit + baseline (~5 min)
.\run.ps1 serve             # start API on :8000
# In a second terminal:
.\run.ps1 ui                # open browser to http://localhost:8501
```

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Sources    │    │  Assembly    │    │  Training    │    │   Serving    │
│              │───▶│              │───▶│              │───▶│              │
│  UCI (858)   │    │ Anchor on    │    │ 18 variants  │    │  FastAPI     │
│  NCBI live   │    │ UCI outcomes │    │ Nested LOOCV │    │  /predict    │
│  1000G       │    │ Provenance   │    │ 5-fold test  │    │  /drift/*    │
│  PGS         │    │ tagged       │    │ Tuned        │    │  Streamlit   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
                                                            ┌──────────────┐
                                                            │  Drift watch │
                                                            │  PSI/KS/χ²   │
                                                            │  Live NCBI   │
                                                            └──────────────┘
```

The integrity contract: **biopsy outcomes are never overwritten by augmentation**. Every value carries a `data_status` tag — `REAL_OBSERVED`, `REAL_COMPUTED`, or `SAMPLED_FROM_PRIOR`. Enforced in `src/storage/provenance.py` and tested by `tests/test_ingestion_smoke.py`.

---

## Modeling

Three feature configurations × six algorithms = 18 variants evaluated.

### Three feature configurations

| Mode | Features | Clinical question |
|---|---|---|
| `uci_only` | 17 UCI risk factors | *Given demographics + history, will biopsy be positive?* (primary screening) |
| `augmented` | UCI + strain + carcinogenicity + PRS + ancestry | *Adding multi-modal molecular features, can we improve screening?* |
| `triage` | UCI + Hinselmann + Schiller + Pap cytology | *Given prior screening tests, will biopsy confirm?* (referral decision — **deployed**) |

### Evaluation protocol — nested

Stratified 80/20 split locked at `random_state=42`. DEV LOOCV with optional inner 3-fold hyperparameter tuning. TEST held out completely; refit on full DEV then 5-fold stratified subsampling for mean ± std. **Test data never participates in any model fit.** Asserted per-fold by `assert_no_leakage()`, verified by `tests/test_no_leakage.py`.

### Results — best variant

`triage + xgb` (tuned).

|  | DEV (LOOCV, n=686) | TEST (5-fold, n=172) |
|---|---|---|
| AUPRC | **65.2%** | **76.7% ± 26.6%** |
| AUROC | **96.7%** | 95.4% ± 7.1% |
| Sensitivity (Youden) | **93.2%** | 80.0% ± 40.0% |
| Specificity (Youden) | 95.3% | 96.3% ± 2.3% |

### Honest negative findings

- **Augmentation does not improve over UCI alone.** HPV strain assignment reaches only 18 of 858 patients (UCI's sparse HPV reporting). Day-4 real PRS doesn't change this on UCI because patient-to-genome matching is arbitrary by construction.
- **IterativeImputer does not improve over SimpleImputer (median).** DEV AUPRC differs by < 1 percentage point across all six algorithms.

Reported transparently because they are themselves engineering deliverables.

---

## API

OpenAPI / Swagger docs at `/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/model/info` | GET | Model name, version, best variant, tuned hyperparameters |
| `/predict/cervical-risk` | POST | Score one patient → probability + tier + audit + SHAP explanation |
| `/drift/baseline` | GET | Saved training-time baseline statistics |
| `/drift/check` | POST | Submit a batch → drift report (PSI / KS per feature) |
| `/drift/strain/live` | GET | Pull live NCBI deposits → compute drift vs published prior (5-min server cache) |

---

## Drift detection

Three statistical tests calibrated to clinical thresholds:

| Method | Detects | Threshold |
|---|---|---|
| **PSI** | Categorical shift | < 0.10 none · 0.10–0.20 minor · ≥ 0.20 significant |
| **KS** 2-sample | Continuous shift | D ≥ 0.10 OR p < 0.05 |
| **Chi-square** | Counts vs expected prior | p < 0.05 |

### Simulated post-vaccination demo (calibrated to Drolet 2019)

| Time | HPV16 share | PSI | Severity | Action |
|---|---|---|---|---|
| Baseline | 55.0% | 0.000 | none | — |
| Year 5 | 50.7% | 0.020 | none | no action |
| Year 10 | 42.1% | 0.156 | minor | monitor |
| Year 20 | 31.2% | 0.495 | significant | **RETRAIN** |

Run: `.\run.ps1 drift`

---

## Frontend

Streamlit clinical decision support UI:

- **Local full version** (`frontend/app.py`) — calls FastAPI over HTTP, includes drift detection + live NCBI auto-refresh. Launch: `.\run.ps1 ui` after `.\run.ps1 serve`.
- **Cloud demo version** (`app.py` at repo root) — single-process, loads the model directly with joblib. Hosted at https://cervirisk-mm.streamlit.app/

Both use:
- Patient summary, risk badge, ranked SHAP feature contributions
- Multi-modal evidence panels (Clinical / Host Genetics / Viral)
- XGBoost built-in TreeSHAP (avoids SHAP-library version coupling)

---

## Tests

```bash
python -m pytest tests/ -v
```

**53 tests across 6 files, all passing.**

| File | Tests | Coverage |
|---|---|---|
| `test_ingestion_smoke.py` | 4 | imports, paths, provenance contract |
| `test_no_leakage.py` | 5 | LOOCV disjointness, leakage detection |
| `test_api.py` | 11 | health, prediction, validation, OpenAPI |
| `test_host_prs.py` | 7 | PRS structure, determinism, populations |
| `test_drift.py` | 17 | PSI/KS/chi-square, baseline, Detector class |
| `test_ncbi_live.py` | 9 | query builder, parsing, caching, errors |

CI runs on every push — see the badge at the top.

---

## Limitations

1. **n = 858 is small.** No model on this cohort matches large-cohort scale.
2. **Augmented mode adds infrastructure, not predictive lift.** Value is the architecture; UCI's HPV reporting caps the metric.
3. **The triage model is not a screening tool.** Assumes colposcopy results exist.
4. **The host PRS is `BIOLOGICALLY_INFORMED_SYNTHETIC`.** Real biology + simulated genotypes; one-line replacement to real VCFs.
5. **Single-site training data.** UCI is one Venezuelan clinic; geographic generalization not validated.

---

## License and data attribution

- **UCI Cervical Cancer Risk Factors:** Fernandes K, Cardoso JS, Fernandes J (2017). CC BY 4.0.
- **NCBI E-utilities:** NCBI Bethesda, MD. Public domain.
- **1000 Genomes Project:** 1000 Genomes Project Consortium. Public domain (EBI/NCBI mirror).
- **PGS Catalog:** Lambert et al., *Nature Genetics* 53:420 (2021). CC0.
- **de Sanjosé HPV prevalence prior:** *Lancet Oncology* 11(11):1048 (2010).
- **Drolet 2019 vaccine impact:** *Lancet* 394:497 (2019). Source for drift simulation parameters.

CerviRisk-MM code: **MIT** (see `LICENSE`).

---

Developed by **Nabras Al-Mahrami** · [nabras.almahrami@ochs.edu.om](mailto:nabras.almahrami@ochs.edu.om) · [github.com/nibrasissa](https://github.com/nibrasissa)
