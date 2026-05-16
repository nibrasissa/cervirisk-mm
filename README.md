# CerviRisk-MM

Multi-modal cervical cancer risk prediction pipeline.

Research prototype, not a medical device.

[![Live demo](https://img.shields.io/badge/live%20demo-cervirisk--mm.streamlit.app-A12F77?style=for-the-badge)](https://cervirisk-mm.streamlit.app/)

[![Tests](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/tests.yml/badge.svg)](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/tests.yml)
[![Docker](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/nibrasissa/cervirisk-mm/actions/workflows/docker-publish.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)

---

## Live demo

https://cervirisk-mm.streamlit.app/

Pick a patient profile in the sidebar, click Run prediction. The Drift detection tab pulls live HPV deposits from NCBI and computes drift against the de Sanjose 2010 prior.

## How to run

Three options.

**Live demo.** Open the link above. No install needed.

**Docker.**

```bash
docker pull ghcr.io/nibrasissa/cervirisk-mm:latest
docker run -p 8000:8000 ghcr.io/nibrasissa/cervirisk-mm:latest
```

API docs at `http://localhost:8000/docs`.

**Clone.**

```powershell
git clone https://github.com/nibrasissa/cervirisk-mm.git
cd cervirisk-mm
.\run.ps1 quickstart   # install, ingest, train, save baseline
.\run.ps1 serve        # API on :8000
.\run.ps1 ui           # UI on :8501 (in a second terminal)
```

On macOS or Linux, use Docker (above) or `make quickstart && make serve && make ui` from the included `Makefile`.

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

## Pipeline stages

Five stages, one module per stage.

**Ingestion** — `src/ingestion/`. One loader per source (UCI, NCBI, 1000 Genomes, PGS). Each writes parquet to `data/raw/`. Loaders fail independently.

**Preprocessing** — Inside the sklearn pipeline. Median imputation for numerics, one-hot for categoricals. Same transformation at training, evaluation, and serving.

**Feature engineering** — `src/features/`. Joins UCI with HPV strain assignment, polygenic risk score, and ancestry tag. Three feature modes: `uci_only`, `augmented`, `triage`.

**Training** — `src/model/train.py`. 18 variants (3 feature modes × 6 algorithms). Stratified 80/20 split, LOOCV on DEV with inner 3-fold tuning, 5-fold stability on held-out TEST. Test data never participates in any fit.

**Inference** — `src/api/main.py` and `src/model/predict.py`. Pydantic validates input, missing fields are imputed. Returns probability, tier, audit, and SHAP attribution. Same code serves the FastAPI endpoint and the cloud demo.

---

## Ingestion cadence

Each data source updates at a different natural rhythm. The pipeline matches each source's cadence rather than forcing them all into the same schedule.

| Source | Cadence | Why this rhythm |
|---|---|---|
| UCI Cervical Cancer Risk Factors | once at training time | Static published dataset (Fernandes 2017). Reingested only when retraining. |
| 1000 Genomes allele frequencies | once at training time | Reference panel, updated by the consortium on a multi-year cycle. |
| PGS Catalog scores | once at training time | Curated, slowly changing. |
| NCBI HPV deposits | **hourly background refresh** | New sequences arrive worldwide every day. The FastAPI service runs a background asyncio task that re-fetches every hour and updates a disk-backed cache. |
| Retraining the model itself | **drift-triggered**, not calendar-triggered | When PSI ≥ 0.20 against the de Sanjose 2010 prior over a sustained window, the drift detector recommends retrain. |

Hourly was chosen for NCBI because it is the slowest rhythm that still feels "live" to a human, any faster would be wasted (NCBI does not change second-to-second), and it stays well within NCBI's polite-use rate limits. The cache lives in memory and on disk (`data/cache/ncbi_hpv_latest.json`) so a restart picks up the last good result rather than starting cold.

Implementation: `src/ingestion/ncbi_scheduler.py`. Tests: `tests/test_ncbi_scheduler.py`.

The cloud demo at https://cervirisk-mm.streamlit.app/ uses a simpler on-demand 5-minute cache (one Streamlit process cannot reliably run background tasks). The Docker image and local `run.ps1 serve` versions use the full hourly scheduler.

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

`run.ps1` maps intent to commands: `ingest`, `fit`, `train`, `serve`, `ui`, `drift`, `test`, `status`, `quickstart`.

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

- Patient records arrive incrementally from a registry and need referential integrity — Postgres with row versioning.
- Predictions support real care decisions and need an immutable audit log — SQLite or Postgres with append-only tables.
- Drift checks run on a schedule and need time-series storage — Postgres with a `drift_check` table.

None of these is true today.

---

## Modeling

### Three feature modes

| Mode | Features | Clinical question |
|---|---|---|
| `uci_only` | 17 UCI risk factors | Given demographics and history, will biopsy be positive? |
| `augmented` | UCI + strain + carcinogenicity + PRS + ancestry | Adding multi-modal features, can we improve? |
| `triage` | UCI + Hinselmann + Schiller + Pap cytology | Given prior screening, will biopsy confirm? (deployed) |

### Results

Best variant: `triage + xgb` (tuned).

|  | DEV (LOOCV, n=686) | TEST (5-fold, n=172) |
|---|---|---|
| AUPRC | 65.2% | 76.7% ± 26.6% |
| AUROC | 96.7% | 95.4% ± 7.1% |
| Sensitivity (Youden) | 93.2% | 80.0% ± 40.0% |
| Specificity (Youden) | 95.3% | 96.3% ± 2.3% |

### What did not work

Augmentation did not beat `uci_only`. HPV strain assignment reached only 18 of 858 patients (UCI has sparse HPV reporting). The real PRS does not help on UCI either, because patient-to-genome matching is arbitrary by construction.

IterativeImputer did not beat median imputation. AUPRC differs by less than one percentage point across all six algorithms.

---

## API

OpenAPI docs at `/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness |
| `/model/info` | GET | Best variant, tuned hyperparameters |
| `/predict/cervical-risk` | POST | Probability, tier, audit, SHAP |
| `/drift/baseline` | GET | Saved baseline statistics |
| `/drift/check` | POST | Drift report on a batch |
| `/drift/strain/live` | GET | Live NCBI strain composition (served from hourly cache) |

---

## Drift detection

| Method | Detects | Threshold |
|---|---|---|
| PSI | Categorical shift | < 0.10 none, 0.10–0.20 minor, ≥ 0.20 significant |
| KS 2-sample | Continuous shift | D ≥ 0.10 or p < 0.05 |
| Chi-square | Counts vs prior | p < 0.05 |

Simulated 20-year vaccination scenario calibrated to Drolet 2019:

| Time | HPV16 share | PSI | Action |
|---|---|---|---|
| Baseline | 55.0% | 0.000 | — |
| Year 5 | 50.7% | 0.020 | no action |
| Year 10 | 42.1% | 0.156 | monitor |
| Year 20 | 31.2% | 0.495 | retrain |

The live demo runs the same detector against current NCBI deposits.

---

## Frontend

Two versions of the same UI.

`frontend/app.py` (local) — calls the FastAPI service over HTTP, with drift auto-refresh. Launch with `.\run.ps1 ui` after `.\run.ps1 serve`.

`app.py` (repo root, used by the live demo) — single process, loads the model directly with joblib. Hosted at https://cervirisk-mm.streamlit.app/.

Both show the patient summary, the risk indicator, ranked SHAP feature contributions, and multi-modal evidence panels.

---

## Tests

```bash
python -m pytest tests/ -v
```

59 tests, all passing.

| File | Tests | Coverage |
|---|---|---|
| `test_ingestion_smoke.py` | 4 | imports, paths, provenance |
| `test_no_leakage.py` | 5 | LOOCV disjointness |
| `test_api.py` | 11 | endpoints, validation |
| `test_host_prs.py` | 7 | PRS structure, determinism |
| `test_drift.py` | 17 | PSI, KS, chi-square |
| `test_ncbi_live.py` | 9 | fetch, parse, cache |
| `test_ncbi_scheduler.py` | 6 | hourly refresh, disk cache, cancellation |

CI runs on every push.

---

## Limitations

- n = 858, single Venezuelan clinic. Geographic generalization is not validated.
- Augmented mode adds infrastructure, not predictive lift.
- The triage model assumes prior colposcopy results. Not a primary screening tool.
- The host PRS uses real GWAS biology but simulated per-individual genotypes.

---

## Data sources

- UCI Cervical Cancer Risk Factors. Fernandes K, Cardoso JS, Fernandes J (2017). CC BY 4.0.
- NCBI E-utilities. Public domain.
- 1000 Genomes Project. Public domain (EBI/NCBI mirror).
- PGS Catalog. Lambert et al., *Nature Genetics* 53:420 (2021). CC0.
- de Sanjose HPV prevalence prior. *Lancet Oncology* 11(11):1048 (2010).
- Drolet 2019 vaccine impact. *Lancet* 394:497 (2019).

---

Developed by Nabras Al-Mahrami. [nabras.almahrami@ochs.edu.om](mailto:nabras.almahrami@ochs.edu.om). [github.com/nibrasissa](https://github.com/nibrasissa).
