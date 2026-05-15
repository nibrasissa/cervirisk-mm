# CerviRisk-MM

**Cervical cancer multi-modal risk prediction  (end-to-end pipeline with continuous data ingestion)**



> **This is a research prototype, not a medically validated tool.** All predictions are tagged `RESEARCH_PROTOTYPE` and accompanied by a clinical disclaimer.

---

## What this project is

A reproducible machine learning pipeline that:

1. **Ingests** four public data sources on demand (UCI clinical, NCBI HPV sequences, 1000 Genomes ancestry, PGS Catalog).
2. **Assembles** synthetic multi-modal patient records by anchoring on real UCI outcomes and layering biologically-informed augmentations, with explicit provenance tracking.
3. **Trains** 18 model variants (3 feature configration × 6 algorithms) under a nested evaluation protocol.
4. **Serves** the best variant through a FastAPI service with three endpoints, dockerized for one-command deployment.

The repo is designed so any reviewer can run `docker compose up` and have a working prediction endpoint in three minutes.

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Data sources    │    │  Feature         │    │  Models          │    │  Serving         │
│                  │    │  engineering     │    │                  │    │                  │
│ UCI (n=858)      │───▶│ Assembly with    │───▶│ 18 variants:     │───▶│ FastAPI:         │
│ NCBI HPV (live)  │    │  provenance      │    │  3 modes ×       │    │  /predict        │
│ 1000 Genomes     │    │ Stable per       │    │  6 algorithms    │    │  /health         │
│ PGS Catalog      │    │  patient_id      │    │ Nested LOOCV     │    │  /model/info     │
│                  │    │                  │    │  + 5-fold test   │    │ Dockerfile       │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │  Drift detection │
                                                │  (days 5–6)      │
                                                │  Retrain trigger │
                                                └──────────────────┘
```

The four sources are independent: there is no public dataset linking host genotype, HPV strain, and clinical outcome for the same patient. CerviRisk-MM solves this with **explicit assembly** rather than fake record joins. The real biopsy outcomes from UCI are anchor points; strain assignments, host PRS, and ancestry matching are augmentations sampled from published priors with provenance tags. The data integrity contract is enforced by `src/storage/provenance.py` and verified by `tests/test_no_leakage.py`.

---

## Quick start

### One-command Docker

```bash
# (one-time) train a model
python -m src.model.train --tune

# boot the API
docker compose up --build

# visit
open http://localhost:8000/docs       # Swagger UI
curl http://localhost:8000/health     # liveness check
```

### Local development

```bash
# 1. Create the conda environment
conda create -n cervirisk python=3.11
conda activate cervirisk
pip install -r requirements.txt

# 2. Verify data sources are reachable
python verify_data_sources.py

# 3. Ingest all four data sources
python -m src.ingestion

# 4. Train and evaluate 18 model variants
python -m src.model.train --tune

# 5. Run the test suite
python -m pytest tests/ -v

# 6. Start the API locally
uvicorn src.api.main:app --reload --port 8000
```

---

## Data sources

| Source | Used for | Records | Provenance |
|---|---|---|---|
| **UCI Cervical Cancer Risk Factors** (id=383) | Anchor — real demographics, history, and biopsy outcomes | 858 patients (55 positives, prevalence 6.4%) | `REAL_OBSERVED` |
| **NCBI nucleotide via E-utilities** | Live HPV sequence feed for strain prevalence drift detection | 160 deposits in last 12 months (HPV16=63, HPV18=6, others) | `REAL_OBSERVED` (sequences); `SAMPLED_FROM_PRIOR` (patient strain assignments) |
| **1000 Genomes phase-3 panel** | Ancestry-matched host individuals for PRS placeholder | 2,504 individuals across 5 super-populations (AMR=347 used for the UCI Caracas cohort) | `REAL_OBSERVED` (panel); `REAL_COMPUTED` (PRS placeholder) |
| **PGS Catalog REST API** | Cervical-cancer-specific polygenic scoring file (best-effort) | 8 candidate scores found; canonical EFO query returned empty | Placeholder for v0.1 — see `docs/DATA_PROVENANCE.md` |

The integrity contract: **biopsy outcomes are never overwritten by augmentation**. Every record carries a `data_status` tag  `REAL_OUTCOME` (from UCI), `SYNTHETIC_ASSEMBLY` (any augmented column derived from a sampled prior), or `REAL_COMPUTED` (deterministic features like strain carcinogenicity). Provenance enforcement is tested with `tests/test_ingestion_smoke.py::test_provenance_protects_biopsy_outcome`.

---

## Modeling

### Three feature configurations

| Mode | Features | Clinical question |
|---|---|---|
| `uci_only` | 17 UCI risk factors | *Given demographics and history, will biopsy be positive?* (primary screening) |
| `augmented` | UCI + assigned HPV strain + strain carcinogenicity + host PRS + ancestry | *Adding multi-modal molecular features, can we improve screening?* |
| `triage` | UCI + Hinselmann + Schiller + Pap cytology | *Given prior screening test results, will biopsy confirm?* (referral decision) |

### Six algorithms

| Model | Why |
|---|---|
| `logreg` | Class-balanced logistic regression, interpretable baseline |
| `logreg_cal` | Same with isotonic calibration, better-calibrated probabilities |
| `rf` | Random forest, class-balanced — non-linear baseline |
| `balanced_rf` | Balanced Random Forest from `imbalanced-learn` undersamples per tree |
| `gbm` | sklearn GradientBoosting with per-sample balanced weights |
| `xgb` | XGBoost — tuned regularization for small-data regime |

### Evaluation protocol — nested

```
858 patients
  └── stratified 80/20 split (random_state=42, locked)
       ├── DEV (686, 44 positives)
       │     ├── Optional hyperparameter tuning (RandomizedSearchCV, 3-fold inner CV, scoring=AUPRC)
       │     ├── LOOCV: 686 out-of-fold predictions per variant
       │     └── Thresholds picked on DEV OOF: {default_0.5, Youden, F1}
       │
       └── TEST (172, 11 positives) — held out completely
             ├── Refit best variant on full DEV, single inference pass
             └── 5-fold stratified subsampling of test predictions
                  → mean ± std for every metric
```

**Test data never participates in any model fit or hyperparameter search.** This is asserted per-fold by `assert_no_leakage()` and verified by `tests/test_no_leakage.py`.

### Metrics computed at every level

Threshold-free: AUROC, AUPRC, Brier score, log loss, prevalence.

Threshold-dependent (at each of 3 thresholds): accuracy, balanced accuracy, sensitivity (recall, TPR), specificity (TNR), precision (PPV), NPV, F1 / F0.5 / F2, MCC, Cohen's kappa, Youden's J, LR+, LR−, diagnostic odds ratio, plus raw confusion matrix counts.

Rate metrics are reported as percentages. See `models/metrics_full.md` for the full block on the chosen variant.

### Results

On the production training run (`--tune --eval nested`), the best variant by DEV AUPRC is **`triage + xgb`**:

| | DEV (LOOCV on 686) | TEST (5-fold on 172) |
|---|---|---|
| AUPRC | **65.2%** | **76.7% ± 26.6%** |
| AUROC | **96.7%** | 95.4% ± 7.1% |
| Sensitivity (Youden) | **93.2%** | 80.0% ± 40.0% |
| Specificity (Youden) | 95.3% | 96.3% ± 2.3% |
| F1 (default) | 73.6% | 86.4% ± 8.2% |

**Honest interpretation.** The triage model performs strongly because its features (Hinselmann acetowhite reaction, Schiller iodine staining, Pap cytology) are *intermediate diagnostic test results that exist before biopsy in the clinical workflow*. They encode most of the disease signal that biopsy is meant to confirm. The model is therefore a triage / referral aid, not a population screening tool. The pre-screening model (`uci_only`) achieves a much more modest DEV AUPRC of 17.4% — consistent with the literature on 858-patient cohorts and the inherent limits of demographic risk factors alone.

The wide test-set standard deviations (especially ±26.6% on AUPRC) reflect the small per-fold positive count (~2 positives per 5-fold chunk).

**Negative results reported honestly:**

- **Augmentation does not improve over UCI alone.** Strain assignment reaches only 18 of 858 patients (2.1%) because UCI's HPV reporting is sparse, and the host PRS is a placeholder. The pipeline architecture accepts real strain genotyping from clinical sources unchanged — but on UCI, augmentation has no measurable effect.
- **IterativeImputer does not improve over SimpleImputer (median).** Tested as a methodological alternative DEV AUPRC differs by <1 percentage point across all six algorithms. The columns with substantial NaN are also the least predictive ones, so no imputer can extract signal that isn't there.

These negative findings are documented for transparency.

---

## API

Three endpoints, OpenAPI/Swagger docs at `/docs`.

### `GET /health`
Liveness probe. Returns `{"status": "ok", "model_loaded": true, "timestamp": "..."}`.

### `GET /model/info`
Loaded model metadata: name, version, tuned hyperparameters, best-variant evaluation summary.

### `POST /predict/cervical-risk`

Request body (all fields optional — imputer handles missingness):

```json
{
  "Age": 38,
  "Number of sexual partners": 4,
  "First sexual intercourse": 16,
  "Num of pregnancies": 2,
  "Smokes": 1,
  "Smokes (years)": 12,
  "Hormonal Contraceptives": 1,
  "Hormonal Contraceptives (years)": 6,
  "IUD": 0,
  "STDs": 1,
  "STDs:HPV": 1,
  "Dx:HPV": 1,
  "Hinselmann": 0,
  "Schiller": 1,
  "Citology": 1
}
```

Response:

```json
{
  "risk_probability": 0.7423,
  "tier": "high",
  "model_version": "cervirisk_mm_v0.1",
  "data_status": "RESEARCH_PROTOTYPE",
  "disclaimer": "This prediction is generated by a research prototype..."
}
```

Tier mapping: `low` for `p < 0.20`, `moderate` for `0.20 ≤ p < 0.50`, `high` otherwise. Tiers are documented and reproducible — patients labeled "high risk" by this model would not in practice receive an immediate diagnostic intervention; the response includes a clinical disclaimer.

---

## Repo layout

```
cervirisk-mm/
├── src/
│   ├── ingestion/        UCI, NCBI, 1000 Genomes, PGS Catalog loaders + orchestrator
│   ├── features/         strain prevalence, host PRS, augmented patient assembly
│   ├── model/            train.py (LOOCV+5-fold), metrics.py, predict.py
│   ├── api/              main.py (FastAPI service)
│   ├── storage/          paths.py (single source of truth), provenance.py
│   └── drift/            (days 5–6 — drift detection module)
├── tests/                20 tests: leakage, immutability, ingestion smoke, API integration
├── data/
│   ├── raw/              parquet outputs from ingestion (UCI, 1KG, NCBI)
│   └── processed/        assembled patient table
├── models/               cervirisk_mm_v0.1.pkl, metrics.json, comparison.md, metrics_full.md, tuned_params.json
├── docs/                 DATA_PROVENANCE.md (the integrity contract)
├── Dockerfile            multi-stage build (builder → runtime)
├── docker-compose.yml    one-command deploy
├── requirements.txt
├── README.md             (this file)
└── verify_data_sources.py
```

---

## Tests

```bash
python -m pytest tests/ -v
```

20 tests across three files:

- `tests/test_ingestion_smoke.py` (4) — imports, paths, provenance contract, data status tagging
- `tests/test_no_leakage.py` (5) — LOOCV index disjointness, stratified split disjointness, leakage detection, strain stability across runs, biopsy outcome immutability
- `tests/test_api.py` (11) — health, root, model info, full prediction, minimal input, empty input, determinism, age validation, binary validation, unknown-field tolerance, OpenAPI schema

All 20 pass on a clean checkout after `python -m src.model.train`.

---

## Roadmap

- Real cervical-cancer PGS from a working PGS Catalog endpoint (or a manually-computed score from Pujol Gualdo et al. 2023 GWAS summary statistics)
- Drift detection module (`src/drift/`): PSI and KS tests on incoming HPV strain distributions vs published priors; retrain trigger
- Live PaVE strain typing for unspecified NCBI records
- SHAP explanations in `/predict/cervical-risk` response
- GitHub Actions CI/CD: lint + tests + build on push; build container to GHCR on release
- Biological drift demonstration: simulate post-vaccination strain replacement, show the drift detector triggering and the model adapting
- Final report polish

---

## Limitations

1. **n=858 is small.** No model on this cohort will reach the AUROC of cervical risk models trained on tens of thousands of patients. Reported metrics are reasonable for the dataset size, not state of the art.
2. **The augmented mode adds infrastructure, not predictive lift.** Strain assignment reaches only 18 patients in UCI. The value is the *architecture* (which accepts real strain genotyping unchanged), not the metrics.
3. **The triage model is not a screening tool.** It assumes the patient is already in colposcopy with intermediate test results. Two complementary clinical models are reported in this repo for clarity.
4. **The host PRS is a placeholder.** Day 4 will integrate a real cervical-cancer scoring file.
5. **Single-site training data.** UCI is one Venezuelan clinic (Caracas). Geographic generalization is not validated.

---

## License & data attribution

UCI Cervical Cancer Risk Factors: K. Fernandes, J. S. Cardoso, J. Fernandes, *Transfer Learning with Partial Observability Applied to Cervical Cancer Screening*, 2017. CC BY 4.0.

NCBI E-utilities: NCBI, Bethesda, MD. Public-domain.

1000 Genomes Project: 1000 Genomes Project Consortium. Public-domain (EBI/NCBI mirror).

PGS Catalog: Lambert et al., *Nature Genetics* 53, 420 (2021). CC0.

de Sanjosé et al. 2010 HPV prevalence priors: *Lancet Oncology* 11(11), 1048. Used as a published reference distribution.

CerviRisk-MM code: MIT.
