# CerviRisk-MM

**Multi-modal cervical cancer risk prediction. End-to-end pipeline with continuous data ingestion.**

> Research prototype, not a medical device. All predictions are tagged `RESEARCH_PROTOTYPE` and accompanied by a clinical disclaimer.

[![Live demo](https://img.shields.io/badge/live%20demo-cervirisk--mm.streamlit.app-A12F77?style=for-the-badge)](https://cervirisk-mm.streamlit.app/)

[![Tests](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/tests.yml/badge.svg)](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/tests.yml)
[![Docker](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/docker-publish.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Try it

Live demo: **https://cervirisk-mm.streamlit.app/**

Click the link, pick a sample patient profile in the sidebar, click **Run prediction**. Returns a risk probability, tier, and SHAP feature contributions explaining the prediction. The Drift detection tab pulls live HPV deposits from NCBI on demand and computes drift against the published de Sanjose 2010 prior.

---

## What this project is

A reproducible machine learning pipeline that:

1. **Ingests** four public data sources on demand (UCI clinical, NCBI HPV sequences, 1000 Genomes ancestry, PGS Catalog).
2. **Assembles** synthetic multi-modal patient records anchored on real UCI outcomes, with explicit provenance tracking.
3. **Trains** 18 model variants (3 feature modes by 6 algorithms) under nested LOOCV plus 5-fold stability evaluation.
4. **Serves** the best variant through a FastAPI service with SHAP feature attribution.
5. **Watches** itself with statistical drift detection (PSI, KS, chi-square) against batch and live NCBI feeds.
6. **Visualizes** everything through a clinical-style Streamlit decision-support UI.

53 automated tests cover every layer. The pipeline is fully reproducible in five minutes on a clean machine.

---

## Three ways to use this project

### 1. Live demo, zero setup

Open https://cervirisk-mm.streamlit.app/ and start clicking. The deployed model is loaded; SHAP attribution works; three sample patients are pre-built; the Drift tab fetches live NCBI data on demand.

### 2. Docker, full version locally without Python install

```bash
docker pull ghcr.io/nibrasissa/cervirisk-mm:latest
docker run -p 8000:8000 ghcr.io/nibrasissa/cervirisk-mm:latest
# Open http://localhost:8000/docs for the Swagger UI
```

### 3. Clone, full pipeline including training

```powershell
git clone https://github.com/nibrasissa/cervirisk-mm.git
cd cervirisk-mm
.\run.ps1 quickstart        # install, ingest, assemble, fit, baseline (~5 min)
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
                                                            │  PSI/KS/chi² │
                                                            │  Live NCBI   │
                                                            └──────────────┘
```

The integrity contract: **biopsy outcomes are never overwritten by augmentation**. Every value carries a `data_status` tag, one of `REAL_OBSERVED`, `REAL_COMPUTED`, or `SAMPLED_FROM_PRIOR`. Enforced in `src/storage/provenance.py` and tested by `tests/test_ingestion_smoke.py`.

---

## Pipeline stages

The five canonical ML pipeline stages, with the responsible module and what each one produces.

### 1. Data ingestion
`src/ingestion/` (one loader per source: `uci.py`, `ncbi.py`, `genomes_1kg.py`, `pgs.py`)

Each loader is independent and writes a tagged parquet to `data/raw/`. Loaders fail independently: if NCBI is down, UCI ingestion still completes. The orchestrator (`python -m src.ingestion`) runs all four and reports per-source status. Live NCBI fetches go through `src/ingestion/ncbi_live.py` with retry and graceful failure.

### 2. Preprocessing
Inside the deployed sklearn Pipeline (`src/model/train.py:build_preprocessor()`)

A `ColumnTransformer` applies `SimpleImputer(strategy='median')` to numeric features and `OneHotEncoder(handle_unknown='ignore')` to categoricals. The preprocessor is part of the saved model artifact, so the exact same transformations apply at training time, evaluation time, and serving time. Missing fields at inference are imputed transparently.

### 3. Feature engineering
`src/features/` (`build_patient.py`, `host_prs.py`, `strain.py`)

Multi-modal assembly that joins UCI outcomes with HPV strain assignment from NCBI prevalence priors, polygenic risk scores computed from eleven published cervical-cancer GWAS variants and 1000 Genomes allele frequencies, and ancestry tagging. Produces `data/processed/patients_assembled.parquet`. Every column carries a provenance tag. Three feature modes are exposed: `uci_only`, `augmented`, `triage`.

### 4. Model training
`src/model/train.py`

Trains 18 variants (3 feature modes by 6 algorithms) under a nested protocol: stratified 80/20 split locked at `random_state=42`, LOOCV on DEV with optional inner 3-fold hyperparameter search, 5-fold stability evaluation on a held-out TEST set. Test data never participates in any model fit, asserted per fold by `assert_no_leakage()`. Outputs `models/cervirisk_mm_v0.1.pkl`, `models/metrics.json`, and `models/tuned_params.json`.

### 5. Inference
`src/api/main.py` plus `src/model/predict.py`

Pydantic schema validates incoming patient records (all fields optional; missing values are imputed by the deployed preprocessor). A single forward pass through the loaded pipeline returns: probability, tier, audit (multi-modal context), and SHAP feature attribution via XGBoost's built-in TreeSHAP. The same code path serves both the FastAPI `/predict/cervical-risk` endpoint and the Streamlit cloud demo at https://cervirisk-mm.streamlit.app/.

---

## Code structure

The codebase is organized **by pipeline layer**, not by file type. Each layer is its own subpackage under `src/`, reads only from upstream layers' artifacts, and writes to its own artifact location.

```
src/
├── ingestion/       fetches raw public data        → data/raw/*.parquet
├── features/        assembles patient table        → data/processed/*.parquet
├── model/           trains, evaluates, predicts    → models/*.pkl + *.json
├── api/             exposes model and drift endpoints
├── drift/           PSI/KS/chi-square detector
└── storage/         provenance + path helpers (cross-cutting)

tests/               mirrors src/ structure: test_ingestion_smoke.py,
                     test_no_leakage.py, test_api.py, test_drift.py, etc.

frontend/            Streamlit clinical decision-support UI (calls API over HTTP)
scripts/             operational scripts (status, demo, predict)
configs/             runtime configuration
docs/                data provenance + design notes
data/                raw and processed parquets (git-ignored, regenerable)
models/              trained artifacts (committed for the live demo)
```

The layering principle is **one-way data flow**: each layer can only depend on layers below it. No cyclic imports. Adding a new data source does not touch the model layer; adding a new algorithm does not touch ingestion. This is what makes "drop in real VCFs for synthetic genotypes" (Day 10 on the roadmap) a one-line change instead of a refactor.

Tests live in `tests/` with one test file per source module. Finding the tests for any module is one-step navigation: `src/drift/detector.py` ↔ `tests/test_drift.py`.

The runner script (`run.ps1`) groups operations by intent: `ingest`, `assemble`, `fit`, `train`, `serve`, `ui`, `drift`, `test`, `status`, `quickstart`. Each command is one verb mapped to one or two `python -m` invocations, so the entire pipeline can be reproduced or inspected with self-documenting commands.

---

## Storage decisions

Storage choices follow the principle: **flat files until a database is genuinely necessary**. n = 858 patients, ~10 MB total artifact set, fully reproducible from public sources. A database would add operational complexity (deploy, backup, schema migrations) without measurable benefit at this scale.

| Stage | Storage | Format | Why this choice |
|---|---|---|---|
| Raw ingestion | flat file | Parquet | Columnar, fast read for tabular data, native pandas roundtrip, smaller than CSV |
| Processed patient table | flat file | Parquet | Same reasons. Reproducible from raw with one command |
| Model artifact | flat file | joblib pickle | sklearn-native serialization with full pipeline state. Versioned (`v0.1`) |
| Metrics and tuned params | flat file | JSON | Human-readable, git-diffable, accessible from any language |
| Drift baseline | flat file | JSON | Same. Loaded at API startup, served from `/drift/baseline` |
| Live NCBI cache | in-memory | Python dict | 5-minute TTL, ephemeral per process. No persistence needed |
| Reports and metric tables | flat file | Markdown | Human-readable, renders in GitHub, version-controlled |

### When this project would move to a database

Three triggers, in order of likelihood:

1. **Patient registry integration.** When ingesting from a hospital registry (Day 9 on the roadmap), records arrive incrementally and need referential integrity. Postgres with row-level versioning.
2. **Prediction logging.** Once predictions support real care decisions, every request needs an immutable audit log. SQLite or Postgres with append-only tables, indexed by patient ID and timestamp.
3. **Drift history.** When drift checks run on a schedule rather than on demand, we need time-series storage. Postgres with a `drift_check` table indexed by timestamp and feature.

Until any of these three exist, flat files are the right choice and the project is more maintainable for it.

### Why Parquet and not CSV?

For the tabular artifacts: Parquet is roughly 5x smaller, 10x faster to read, preserves dtypes including dates and categoricals, and supports column-pruned reads. The only CSV input is the UCI download itself, which is converted to Parquet at ingestion time.

### Why JSON and not YAML for configuration?

For metrics, baselines, and tuned hyperparameters: JSON is unambiguous (no implicit type coercion), every language reads it, it diffs cleanly in git, and `pandas` / `Pydantic` parse it natively. YAML's quoting rules would be a recurring source of bugs for the kinds of mixed numeric and string data these files contain.

---

## Modeling

Three feature configurations times six algorithms equals 18 variants evaluated.

### Three feature configurations

| Mode | Features | Clinical question |
|---|---|---|
| `uci_only` | 17 UCI risk factors | Given demographics and history, will biopsy be positive? (primary screening) |
| `augmented` | UCI + strain + carcinogenicity + PRS + ancestry | Adding multi-modal molecular features, can we improve screening? |
| `triage` | UCI + Hinselmann + Schiller + Pap cytology | Given prior screening tests, will biopsy confirm? (referral decision, **deployed**) |

### Evaluation protocol

Stratified 80/20 split locked at `random_state=42`. DEV LOOCV with optional inner 3-fold hyperparameter tuning. TEST held out completely; refit on full DEV then 5-fold stratified subsampling for mean ± std. Test data never participates in any model fit, asserted per-fold by `assert_no_leakage()`, verified by `tests/test_no_leakage.py`.

### Results

Best variant: `triage + xgb` (tuned).

|  | DEV (LOOCV, n=686) | TEST (5-fold, n=172) |
|---|---|---|
| AUPRC | **65.2%** | **76.7% ± 26.6%** |
| AUROC | **96.7%** | 95.4% ± 7.1% |
| Sensitivity (Youden) | **93.2%** | 80.0% ± 40.0% |
| Specificity (Youden) | 95.3% | 96.3% ± 2.3% |

### Honest negative findings

- **Augmentation does not improve over UCI alone.** HPV strain assignment reaches only 18 of 858 patients (UCI's sparse HPV reporting). Day-4 real PRS does not change this on UCI because patient-to-genome matching is arbitrary by construction.
- **IterativeImputer does not improve over SimpleImputer (median).** DEV AUPRC differs by less than one percentage point across all six algorithms.

Reported transparently because they are themselves engineering deliverables.

---

## API

OpenAPI / Swagger docs at `/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/model/info` | GET | Model name, version, best variant, tuned hyperparameters |
| `/predict/cervical-risk` | POST | Score one patient. Returns probability, tier, audit, and SHAP explanation |
| `/drift/baseline` | GET | Saved training-time baseline statistics |
| `/drift/check` | POST | Submit a batch. Returns drift report (PSI and KS per feature) |
| `/drift/strain/live` | GET | Pull live NCBI deposits, compute drift vs published prior (5-min server cache) |

---

## Drift detection

Three statistical tests calibrated to clinical thresholds:

| Method | Detects | Threshold |
|---|---|---|
| PSI | Categorical shift | < 0.10 none, 0.10–0.20 minor, ≥ 0.20 significant |
| KS 2-sample | Continuous shift | D ≥ 0.10 or p < 0.05 |
| Chi-square | Counts vs expected prior | p < 0.05 |

### Simulated post-vaccination demo (calibrated to Drolet 2019)

| Time | HPV16 share | PSI | Severity | Action |
|---|---|---|---|---|
| Baseline | 55.0% | 0.000 | none | — |
| Year 5 | 50.7% | 0.020 | none | no action |
| Year 10 | 42.1% | 0.156 | minor | monitor |
| Year 20 | 31.2% | 0.495 | significant | **retrain** |

Run: `.\run.ps1 drift`

The live demo at https://cervirisk-mm.streamlit.app/ performs the same drift computation against current NCBI deposits, on demand.

---

## Frontend

Streamlit clinical decision support UI:

- **Local full version** (`frontend/app.py`): calls FastAPI over HTTP, includes drift detection plus live NCBI auto-refresh. Launch with `.\run.ps1 ui` after `.\run.ps1 serve`.
- **Cloud demo version** (`app.py` at repo root): single-process, loads the model directly with joblib. Hosted at https://cervirisk-mm.streamlit.app/

Both use:
- Patient summary, risk indicator, ranked SHAP feature contributions
- Multi-modal evidence panels (clinical, host genetics, viral)
- XGBoost built-in TreeSHAP, which avoids SHAP-library version coupling

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
| `test_drift.py` | 17 | PSI, KS, chi-square, baseline, Detector class |
| `test_ncbi_live.py` | 9 | query builder, parsing, caching, errors |

CI runs on every push; see the badge at the top.

---

## Limitations

1. **n = 858 is small.** No model on this cohort matches large-cohort scale.
2. **Augmented mode adds infrastructure, not predictive lift.** Value is the architecture; UCI's HPV reporting caps the metric.
3. **The triage model is not a screening tool.** It assumes colposcopy results exist.
4. **The host PRS is `BIOLOGICALLY_INFORMED_SYNTHETIC`.** Real biology, simulated genotypes; one-line replacement to real VCFs.
5. **Single-site training data.** UCI is one Venezuelan clinic; geographic generalization is not validated.

---

## License and data attribution

- UCI Cervical Cancer Risk Factors: Fernandes K, Cardoso JS, Fernandes J (2017). CC BY 4.0.
- NCBI E-utilities: NCBI Bethesda, MD. Public domain.
- 1000 Genomes Project: 1000 Genomes Project Consortium. Public domain (EBI/NCBI mirror).
- PGS Catalog: Lambert et al., *Nature Genetics* 53:420 (2021). CC0.
- de Sanjose HPV prevalence prior: *Lancet Oncology* 11(11):1048 (2010).
- Drolet 2019 vaccine impact: *Lancet* 394:497 (2019). Source for drift simulation parameters.

CerviRisk-MM code: MIT (see `LICENSE`).

---

Developed by **Nabras Al-Mahrami**. [nabras.almahrami@ochs.edu.om](mailto:nabras.almahrami@ochs.edu.om). [github.com/nibrasissa](https://github.com/nibrasissa).
