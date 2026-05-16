# CerviRisk-MM

Multi-modal cervical cancer risk prediction pipeline.

Research prototype, not a medical device.

[![Live demo](https://img.shields.io/badge/live%20demo-cervirisk--mm.streamlit.app-A12F77?style=for-the-badge)](https://cervirisk-mm.streamlit.app/)

[![Tests](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/tests.yml/badge.svg)](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/tests.yml)
[![Docker](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/docker-publish.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)

---

## Overview

CerviRisk-MM is an end-to-end machine learning pipeline for cervical cancer risk prediction. It demonstrates production-oriented engineering across the full pipeline: continuous data ingestion from four public sources, multi-modal feature assembly, leak-safe training, FastAPI serving with SHAP explanations, real-time drift monitoring against published epidemiological priors, and a Streamlit dashboard for clinician interaction.

The project deliberately separates **monitoring** (which uses live HPV strain composition from NCBI) from **prediction** (which uses patient clinical features). This separation is the engineering thesis of the project: a production ML system must watch its inputs even when the model itself does not depend on them.

---

## Live demo

https://cervirisk-mm.streamlit.app/

Three tabs. The Drift detection tab fetches live HPV deposits from the NCBI Entrez API on every page load, computes PSI against the de Sanjose 2010 prior, decomposes the drift signal per strain, and emits a recommended action.

---

## How to run

Pick whichever fits your environment.

### Option 1 — Live demo (no install)

Open https://cervirisk-mm.streamlit.app/ in any browser.

Three tabs: Predict, Drift detection, About this model.

### Option 2 — Docker (API only)

Runs the FastAPI service so a reviewer can inspect endpoints, run predictions, and exercise the drift API in a reproducible container.

```bash
docker pull ghcr.io/nibrasissa/cervirisk-mm:latest
docker run -p 8000:8000 ghcr.io/nibrasissa/cervirisk-mm:latest
```

Open `http://localhost:8000/docs` for the Swagger UI.

For the interactive dashboard, use the live demo above or run locally (Option 3).

### Option 3 — Clone and run locally (API + dashboard)

Requires Python 3.11.

```bash
git clone https://github.com/nibrasissa/cervirisk-mm.git
cd cervirisk-mm
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell:  `.\.venv\Scripts\Activate.ps1`
- macOS / Linux:       `source .venv/bin/activate`

Install dependencies and verify tests:

```bash
pip install -r requirements.txt
python -m pytest tests/ -v         # 59 tests, ~15 seconds
```

Start the API in one terminal:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Start the dashboard in a second terminal:

```bash
streamlit run app.py --server.port 8501
```

Open in your browser:

- `http://localhost:8501` — Streamlit dashboard
- `http://localhost:8000/docs` — FastAPI Swagger documentation

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

Biopsy outcomes are never overwritten by feature augmentation. Every value carries a `data_status` tag: `REAL_OBSERVED`, `REAL_COMPUTED`, or `SAMPLED_FROM_PRIOR`. Enforced in `src/storage/provenance.py`.

---

## Results

Best variant: **`triage + xgb` (tuned)**. The deployed model.

### Headline metrics

| Metric | DEV (LOOCV, n=686) | TEST (5-fold, n=172) |
|---|---|---|
| AUPRC | **65.2 %** | 76.7 % ± 26.6 % |
| AUROC | **96.7 %** | 95.4 % ± 7.1 % |
| Sensitivity (Youden) | 93.2 % | 80.0 % ± 40.0 % |
| Specificity (Youden) | 95.3 % | 96.3 % ± 2.3 % |

DEV metrics come from leave-one-out cross-validation on 686 patients with 44 biopsy positives. TEST metrics come from 5-fold stratified subsampling on the held-out 172-patient set (11 positives), reported with standard deviation. The wide TEST sensitivity band reflects ~2 positives per 5-fold chunk.

### How 18 variants narrowed to one

Trained over three feature modes × six algorithms:

| Mode | Features | Best DEV AUPRC |
|---|---|---|
| `uci_only` | 17 UCI risk factors | 27.4 % |
| `augmented` | UCI + HPV strain + carcinogenicity + PRS + ancestry | 24.0 % |
| `triage` | UCI + Hinselmann + Schiller + Pap cytology | **65.2 %** |

The triage variant won decisively. Adding strain and host genetics did not improve over UCI alone. The full comparison is in `models/comparison.md`.

### What did not work

Augmentation did not beat `uci_only`. HPV strain assignment reached only 18 of 858 patients (UCI has sparse HPV reporting). The real PRS does not help on UCI either, because patient-to-genome matching is arbitrary by construction.

IterativeImputer did not beat median imputation. AUPRC differs by less than one percentage point across all six algorithms.

---

## Pipeline stages

Five stages, one module per stage.

**Ingestion** — `src/ingestion/`. One loader per source (UCI, NCBI, 1000 Genomes, PGS). Each writes parquet to `data/raw/`. Loaders fail independently.

**Preprocessing** — Inside the sklearn pipeline. Median imputation for numerics, one-hot for categoricals. Same transformation at training, evaluation, and serving.

**Feature engineering** — `src/features/`. Joins UCI with HPV strain assignment, polygenic risk score, and ancestry tag.

**Training** — `src/model/train.py`. 18 variants, stratified 80/20 split, LOOCV on DEV with inner 3-fold tuning, 5-fold stability on held-out TEST. Test data never participates in any fit.

**Inference** — `src/api/main.py` and `src/model/predict.py`. Pydantic validates input, missing fields are imputed. Returns probability, tier, audit, and SHAP attribution.

---

## Ingestion cadence

Each data source updates at a different natural rhythm.

| Source | Cadence | Why this rhythm |
|---|---|---|
| UCI Cervical Cancer Risk Factors | once at training time | Static published dataset (Fernandes 2017). Reingested only when retraining. |
| 1000 Genomes allele frequencies | once at training time | Reference panel, updated by the consortium on a multi-year cycle. |
| PGS Catalog scores | once at training time | Curated, slowly changing. |
| NCBI HPV deposits | **hourly background refresh** | New sequences arrive worldwide every day. The FastAPI service runs a background asyncio task that re-fetches every hour and updates a disk-backed cache at `data/cache/ncbi_hpv_latest.json`. |
| Retraining the model | **drift-triggered**, not calendar-triggered | When PSI ≥ 0.20 against the de Sanjose 2010 prior over a sustained window, the drift detector recommends retrain. |

Hourly was chosen for NCBI because it is the slowest rhythm that still feels live to a human and stays well within NCBI's polite-use rate limits.

Implementation: `src/ingestion/ncbi_scheduler.py`. Tests: `tests/test_ncbi_scheduler.py`.

The cloud demo at https://cervirisk-mm.streamlit.app/ uses a simpler on-demand 5-minute cache because one Streamlit process cannot reliably run background tasks. The Docker image and local `uvicorn` deployment use the full hourly scheduler.

---

## Code structure

Organized by pipeline layer, not by file type.

```
src/
├── ingestion/      fetches public data       → data/raw/*.parquet
├── features/       assembles patient table   → data/processed/*.parquet
├── model/          trains, predicts          → models/*.pkl + *.json
├── api/            FastAPI endpoints
├── drift/          PSI / KS / chi-square detector
└── storage/        provenance + paths

tests/              one test file per source module
frontend/           Streamlit UI (calls API)
scripts/            status, demo, predict
models/             trained artifacts (committed for the demo)
data/               raw + processed parquets (regenerable, git-ignored)
```

Each layer reads only from layers below it. No cyclic imports. Adding a new data source does not touch the model layer; adding a new algorithm does not touch ingestion.

---

## Storage decisions

Flat files. n = 858 patients, total artifact set under 10 MB, fully regenerable from public sources. A database adds operational complexity without measurable benefit at this scale.

| Stage | Format | Why |
|---|---|---|
| Raw ingestion | Parquet | Columnar, fast, smaller than CSV, native pandas |
| Processed table | Parquet | Same |
| Model artifact | joblib pickle | sklearn-native, full pipeline state |
| Metrics, baseline, hyperparameters | JSON | Human-readable, git-diffable |
| Live NCBI cache | in-memory dict + JSON on disk | Hourly refresh, restart-safe |
| Reports | Markdown | Renders in GitHub |

A database would make sense in three scenarios:

- Patient records arrive incrementally from a registry and need referential integrity → Postgres with row versioning.
- Predictions support real care decisions and need an immutable audit log → SQLite or Postgres with append-only tables.
- Drift checks run on a schedule and need time-series storage → Postgres with a `drift_check` table.

None of these is true today.

---

## API

OpenAPI documentation at `/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/model/info` | GET | Best variant, tuned hyperparameters |
| `/predict/cervical-risk` | POST | Probability, tier, audit, SHAP attribution |
| `/drift/baseline` | GET | Saved baseline statistics |
| `/drift/check` | POST | Drift report on a batch |
| `/drift/strain/live` | GET | Live NCBI strain composition (served from hourly cache) |

---

## Drift detection

The system uses the Population Stability Index (PSI) on categorical strain composition to detect shifts between live NCBI deposits and the de Sanjose 2010 cervical-cancer prevalence prior.

| Metric | Detects | Threshold |
|---|---|---|
| PSI | Shift in HPV strain composition vs the published prior | < 0.10 none, 0.10–0.20 minor, ≥ 0.20 significant |

The dashboard answers the four monitoring questions:

1. **What changed?** — PSI card with severity classification.
2. **By how much?** — Numeric PSI value and severity (none / minor / significant).
3. **Where in the data?** — Per-strain PSI decomposition. Each strain's contribution to total PSI is shown as a ranked horizontal bar, with direction (over- or under-represented). The sum of contributions equals total PSI.
4. **What do we do?** — Recommended action block with trigger threshold, current PSI, largest driver, concrete next step, and owner contact.

Simulated 20-year vaccination scenario calibrated to Drolet 2019 (Lancet meta-analysis of 65 studies):

| Time | HPV16 share | PSI | Action |
|---|---|---|---|
| Baseline | 55.0 % | 0.000 | — |
| Year 5 | 50.7 % | 0.020 | no action |
| Year 10 | 42.1 % | 0.156 | monitor |
| Year 20 | 31.2 % | 0.495 | retrain |

The live demo runs the same detector against current NCBI deposits.

---

## Tests

```bash
python -m pytest tests/ -v
```

**59 tests, all passing**, run on every push via GitHub Actions.

| File | Tests | Coverage |
|---|---|---|
| `test_ingestion_smoke.py` | 4 | imports, paths, provenance |
| `test_no_leakage.py` | 5 | LOOCV disjointness, outcome preservation |
| `test_api.py` | 11 | endpoints, validation, determinism |
| `test_host_prs.py` | 7 | PRS structure, determinism, ancestry stratification |
| `test_drift.py` | 17 | PSI computation, severity classification, baseline persistence |
| `test_ncbi_live.py` | 9 | fetch, parse, retry, cache |
| `test_ncbi_scheduler.py` | 6 | hourly refresh, disk cache, cancellation |

---

## Limitations

- n = 858, single Venezuelan clinic. Geographic generalization is not validated.
- The deployed `triage + xgb` model assumes prior colposcopy results (Hinselmann, Schiller, cytology). Not a primary screening tool.
- The host PRS uses real GWAS biology but simulated per-individual genotypes (`BIOLOGICALLY_INFORMED_SYNTHETIC`).
- Augmented mode adds multi-modal infrastructure but no predictive lift on this cohort.

---

## Data sources

- UCI Cervical Cancer Risk Factors. Fernandes K, Cardoso JS, Fernandes J (2017). CC BY 4.0.
- NCBI E-utilities. Public domain.
- 1000 Genomes Project. Public domain (EBI/NCBI mirror).
- PGS Catalog. Lambert et al., *Nature Genetics* 53:420 (2021). CC0.
- de Sanjose HPV prevalence prior. *Lancet Oncology* 11(11):1048 (2010).
- Drolet vaccine impact. *Lancet* 394:497 (2019).

---

Developed by Nabras Al-Mahrami. [nabras.almahrami@ochs.edu.om](mailto:nabras.almahrami@ochs.edu.om). [github.com/nibrasissa](https://github.com/nibrasissa).
