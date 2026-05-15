"""CerviRisk-MM FastAPI service.

Endpoints
---------
GET  /health              — liveness probe, returns 200 if the process is up
GET  /model/info          — metadata about the loaded model: name, version,
                            tuned hyperparameters, evaluation summary
POST /predict/cervical-risk — score one patient, return risk probability,
                            tier (low/moderate/high), and a clinical disclaimer
GET  /drift/baseline      — return the saved baseline statistics
POST /drift/check         — submit a batch of records, get a drift report
                            against the saved baseline
GET  /drift/strain/live   — pull last 30 days of HPV deposits from NCBI
                            E-utilities and compute drift vs the published
                            de Sanjosé 2010 prior (5-min cache)
GET  /docs                — auto-generated OpenAPI / Swagger UI
GET  /                    — minimal landing page

Design notes
------------
- Pydantic v2 models for request/response schemas, so /docs is auto-populated.
- All UCI input fields are optional; missing values flow through the trained
  pipeline's imputer (median by default, or IterativeImputer if the artifact
  was trained with it). This matches the real-world case where intake forms
  are partially filled.
- The model artifact is loaded ONCE at process start (not per request).
- The response includes a `data_status` field marking outputs as research-only.
- No PHI is logged. Only request structure and outcome are emitted.

The trained pipeline knows which features it expects via the saved
ColumnTransformer; we pass a superset and let it select.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.storage.paths import MODELS_DIR

logger = logging.getLogger("cervirisk.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


# -------------------------------------------------------------------------
# Globals — loaded once at startup
# -------------------------------------------------------------------------
_state: dict[str, Any] = {
    "model": None,
    "model_path": None,
    "model_loaded_at": None,
    "metrics": None,
    "tuned_params": None,
    "drift_baseline": None,
    "drift_detector": None,
}


def _load_model() -> None:
    """Load the trained pipeline, metric metadata, and drift baseline."""
    model_path = MODELS_DIR / "cervirisk_mm_v0.1.pkl"
    if not model_path.exists():
        logger.warning(
            "Model artifact not found at %s — endpoints will return 503 "
            "until `python -m src.model.train` is run.", model_path
        )
        return

    _state["model"] = joblib.load(model_path)
    _state["model_path"] = str(model_path)
    _state["model_loaded_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("Loaded model from %s", model_path)

    # Optional metadata — best-effort
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            _state["metrics"] = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load metrics.json: %s", e)

    params_path = MODELS_DIR / "tuned_params.json"
    if params_path.exists():
        try:
            _state["tuned_params"] = json.loads(params_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load tuned_params.json: %s", e)

    # Drift baseline — also best-effort
    try:
        from src.drift import load_baseline, DriftDetector
        baseline = load_baseline()
        if baseline:
            _state["drift_baseline"] = baseline
            _state["drift_detector"] = DriftDetector(baseline)
            logger.info("Loaded drift baseline (%d numeric, %d categorical features)",
                        len(baseline.get("numeric_features", {})),
                        len(baseline.get("categorical_features", {})))
    except Exception as e:
        logger.warning("Could not load drift baseline (non-fatal): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook — loads model at boot."""
    _load_model()
    yield


# -------------------------------------------------------------------------
# Pydantic schemas
# -------------------------------------------------------------------------
class PatientFeatures(BaseModel):
    """All UCI features plus optional augmented features.

    Every field is optional so partially-filled intake forms are accepted;
    the model's imputer handles missingness.
    """
    # UCI risk factors
    age: float | None = Field(None, ge=10, le=110, alias="Age")
    number_of_sexual_partners: float | None = Field(None, ge=0, alias="Number of sexual partners")
    first_sexual_intercourse: float | None = Field(None, ge=8, le=80, alias="First sexual intercourse")
    num_of_pregnancies: float | None = Field(None, ge=0, alias="Num of pregnancies")
    smokes: float | None = Field(None, ge=0, le=1, alias="Smokes")
    smokes_years: float | None = Field(None, ge=0, alias="Smokes (years)")
    hormonal_contraceptives: float | None = Field(None, ge=0, le=1, alias="Hormonal Contraceptives")
    hormonal_contraceptives_years: float | None = Field(None, ge=0, alias="Hormonal Contraceptives (years)")
    iud: float | None = Field(None, ge=0, le=1, alias="IUD")
    iud_years: float | None = Field(None, ge=0, alias="IUD (years)")
    stds: float | None = Field(None, ge=0, le=1, alias="STDs")
    stds_number: float | None = Field(None, ge=0, alias="STDs (number)")
    stds_hpv: float | None = Field(None, ge=0, le=1, alias="STDs:HPV")
    stds_hiv: float | None = Field(None, ge=0, le=1, alias="STDs:HIV")
    dx_cancer: float | None = Field(None, ge=0, le=1, alias="Dx:Cancer")
    dx_cin: float | None = Field(None, ge=0, le=1, alias="Dx:CIN")
    dx_hpv: float | None = Field(None, ge=0, le=1, alias="Dx:HPV")

    # Augmented features (used by augmented-mode models)
    assigned_hpv_strain: str | None = Field(
        None, description="HPV strain key, e.g. HPV16, HPV18, NONE"
    )
    strain_carcinogenicity: float | None = Field(None, ge=0, le=1)
    host_prs: float | None = None
    matched_super_pop: str | None = Field(None, description="1000G super-population code")

    # Triage features (used by triage-mode model)
    hinselmann: float | None = Field(None, ge=0, le=1, alias="Hinselmann")
    schiller: float | None = Field(None, ge=0, le=1, alias="Schiller")
    citology: float | None = Field(None, ge=0, le=1, alias="Citology")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., ge=0, le=1)
    tier: str = Field(..., description="low | moderate | high")
    model_version: str
    data_status: str
    disclaimer: str
    audit: dict | None = Field(
        None,
        description=(
            "Multi-modal context the model used: host genetics (PRS, ancestry), "
            "HPV viral (strain, carcinogenicity), clinical aggregate."
        ),
    )
    explanation: dict | None = Field(
        None,
        description=(
            "Top feature contributions to this prediction (SHAP TreeExplainer "
            "on tree-based variants). May be null if SHAP unavailable."
        ),
    )


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_path: str | None
    model_loaded_at: str | None
    best_variant: dict | None
    tuned_hyperparameters: dict | None
    notes: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class DriftCheckRequest(BaseModel):
    """A batch of patient records to check against the saved baseline.

    Each record is a dict of feature name → value. Field names match the
    PatientFeatures schema (the aliased UCI column names like 'Age',
    'STDs:HPV', etc.).
    """
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=10000)


class DriftCheckResponse(BaseModel):
    drift_detected: bool
    n_features_checked: int
    n_features_with_drift: int
    severity_summary: dict[str, int]
    action_recommended: str
    by_feature: list[dict[str, Any]]
    baseline_label: str | None
    checked_at: str


# -------------------------------------------------------------------------
# App
# -------------------------------------------------------------------------
app = FastAPI(
    title="CerviRisk-MM",
    description=(
        "Cervical cancer multi-modal risk prediction. "
        "Trained on UCI Cervical Cancer Risk Factors (n=858) with biological "
        "augmentations from NCBI HPV sequences, 1000 Genomes ancestry "
        "matching, and PGS Catalog scoring files. "
        "Research prototype — not a medical device."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
TIER_THRESHOLDS = {"low_max": 0.20, "moderate_max": 0.50}

DISCLAIMER = (
    "This prediction is generated by a research prototype trained on the UCI "
    "Cervical Cancer Risk Factors dataset (n=858) with biological augmentations "
    "from public sources. It is not a medical diagnosis. See "
    "docs/DATA_PROVENANCE.md for the full integrity contract."
)


def _tier_for(p: float) -> str:
    if p < TIER_THRESHOLDS["low_max"]:
        return "low"
    if p < TIER_THRESHOLDS["moderate_max"]:
        return "moderate"
    return "high"


def _features_to_row(features: PatientFeatures) -> pd.DataFrame:
    """Convert Pydantic model into a single-row DataFrame with UCI column names.

    Uses the aliases on the Pydantic model so that the DataFrame columns
    match what the trained sklearn pipeline expects.
    """
    raw = features.model_dump(by_alias=True, exclude_none=False)
    # Convert None -> NaN so pandas treats missingness consistently.
    cleaned = {k: (None if v is None else v) for k, v in raw.items()}
    return pd.DataFrame([cleaned])


def _best_variant_info() -> dict | None:
    """Return the best variant from metrics.json (by DEV AUPRC at Youden)."""
    if not _state.get("metrics"):
        return None

    def score(r: dict) -> float:
        dev = r.get("dev") or {}
        if isinstance(dev, dict):
            v = (dev.get("youden") or {}).get("auprc")
            if v is not None:
                return v
        return 0.0

    best = max(_state["metrics"], key=score)
    dev = (best.get("dev") or {}).get("youden", {}) if isinstance(best.get("dev"), dict) else {}
    test = (best.get("test") or {}).get("youden", {}) if isinstance(best.get("test"), dict) else {}

    return {
        "mode": best.get("mode"),
        "model": best.get("model"),
        "eval": best.get("eval"),
        "dev_auprc_pct": round(100 * dev.get("auprc", 0), 1) if dev else None,
        "dev_auroc_pct": round(100 * dev.get("auroc", 0), 1) if dev else None,
        "dev_sensitivity_pct": round(100 * dev.get("sensitivity", 0), 1) if dev else None,
        "dev_specificity_pct": round(100 * dev.get("specificity", 0), 1) if dev else None,
        "test_auprc_pct": round(100 * test.get("auprc_mean", 0), 1) if test else None,
        "test_auprc_std_pct": round(100 * test.get("auprc_std", 0), 1) if test else None,
    }


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Minimal landing — points to /docs."""
    return {
        "service": "CerviRisk-MM v0.1",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict/cervical-risk",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("model") is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if _state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.model.train` first.",
        )
    return ModelInfoResponse(
        model_name="cervirisk_mm",
        model_version="0.1.0",
        model_path=_state.get("model_path"),
        model_loaded_at=_state.get("model_loaded_at"),
        best_variant=_best_variant_info(),
        tuned_hyperparameters=_state.get("tuned_params"),
        notes=[
            "Research prototype — not a medical device.",
            "Trained on UCI Cervical Cancer Risk Factors (n=858).",
            "Nested evaluation: LOOCV on DEV (80%, n=686) + 5-fold stability "
            "on held-out TEST (20%, n=172).",
            "Three feature modes available: uci_only, augmented, triage.",
            "Deployed model uses the best-by-DEV-AUPRC variant.",
        ],
    )


@app.post("/predict/cervical-risk", response_model=PredictionResponse)
def predict_cervical_risk(features: PatientFeatures) -> PredictionResponse:
    if _state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.model.train` first.",
        )

    df = _features_to_row(features)

    try:
        proba = float(_state["model"].predict_proba(df)[0, 1])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {type(e).__name__}: {e}",
        )

    logger.info("Predicted risk_probability=%.4f tier=%s", proba, _tier_for(proba))

    # Multi-modal audit + per-prediction explanation (both graceful fallback)
    audit = None
    explanation = None
    try:
        from src.model.explain import build_audit, explain_prediction
        patient_dict = features.model_dump(by_alias=True, exclude_none=False)
        audit = build_audit(patient_dict, baseline=_state.get("drift_baseline"))
        explanation = explain_prediction(_state["model"], df, top_k=10)
    except Exception as e:
        logger.warning("Audit/explanation failed (non-fatal): %s", e)

    return PredictionResponse(
        risk_probability=round(proba, 4),
        tier=_tier_for(proba),
        model_version="cervirisk_mm_v0.1",
        data_status="RESEARCH_PROTOTYPE",
        disclaimer=DISCLAIMER,
        audit=audit,
        explanation=explanation,
    )


# -------------------------------------------------------------------------
# Drift endpoints
# -------------------------------------------------------------------------
@app.get("/drift/baseline")
def drift_baseline() -> dict[str, Any]:
    """Return a summary of the saved drift baseline."""
    baseline = _state.get("drift_baseline")
    if baseline is None:
        raise HTTPException(
            status_code=503,
            detail="Drift baseline not loaded. Run "
                   "`python -m src.drift.baseline` first."
        )
    # Return a compact summary, not the full value samples (those bloat the response)
    numeric_summary = {
        col: {k: v for k, v in info.items() if k != "values_sample"}
        for col, info in baseline.get("numeric_features", {}).items()
    }
    return {
        "version": baseline.get("version"),
        "captured_at": baseline.get("captured_at"),
        "source_label": baseline.get("source_label"),
        "n_records": baseline.get("n_records"),
        "numeric_features": numeric_summary,
        "categorical_features": baseline.get("categorical_features", {}),
    }


@app.post("/drift/check", response_model=DriftCheckResponse)
def drift_check(body: DriftCheckRequest) -> DriftCheckResponse:
    """Compare a batch of records against the saved baseline.

    Each record is a dict (same shape as POST /predict/cervical-risk). The
    detector runs PSI on categorical features and KS-2-sample on numeric
    features, against the saved training-time baseline.

    Returns per-feature drift scores plus an overall recommendation:
        - 'no_action'  no drift in any feature
        - 'monitor'    minor drift, watch the next batch
        - 'retrain'    significant drift on at least one feature
    """
    detector = _state.get("drift_detector")
    baseline = _state.get("drift_baseline")
    if detector is None or baseline is None:
        raise HTTPException(
            status_code=503,
            detail="Drift baseline not loaded. Run "
                   "`python -m src.drift.baseline` first."
        )

    df = pd.DataFrame(body.records)
    if df.empty:
        raise HTTPException(status_code=400, detail="No records provided.")

    try:
        report = detector.scan(df)
    except Exception as e:
        logger.exception("Drift scan failed")
        raise HTTPException(
            status_code=400,
            detail=f"Drift scan failed: {type(e).__name__}: {e}",
        )

    logger.info("Drift check: %d/%d features drifted, action=%s",
                report["n_features_with_drift"],
                report["n_features_checked"],
                report["action_recommended"])

    return DriftCheckResponse(
        drift_detected=report["drift_detected"],
        n_features_checked=report["n_features_checked"],
        n_features_with_drift=report["n_features_with_drift"],
        severity_summary=report["severity_summary"],
        action_recommended=report["action_recommended"],
        by_feature=report["by_feature"],
        baseline_label=baseline.get("source_label"),
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


# -------------------------------------------------------------------------
# Live NCBI drift — direct fetch with 5-minute cache to be gentle on NCBI
# -------------------------------------------------------------------------
import threading
from collections import Counter as _Counter

_ncbi_live_cache: dict[str, Any] = {
    "fetched_at": None,
    "ncbi_meta": None,
    "result": None,
    "ttl_seconds": 300,   # 5 minutes — even if frontend polls every minute
}
_ncbi_live_lock = threading.Lock()


def _compute_live_drift_payload(days: int) -> dict[str, Any]:
    """Pull NCBI, compute strain distribution + PSI + chi-square vs baseline."""
    from src.drift import psi, psi_per_category, psi_severity, chi_square_fit
    from src.features.strain import (
        PUBLISHED_PREVALENCE,
        collapse_to_categories,
        extract_strain_from_title,
    )
    from src.ingestion.ncbi_live import fetch_recent_deposits

    df, meta = fetch_recent_deposits(days=days)

    payload: dict[str, Any] = {
        "ncbi_meta": meta,
        "days_window": days,
        "baseline_distribution": dict(PUBLISHED_PREVALENCE),
        "current_distribution": {},
        "n_sequences_typed": 0,
        "psi_score": 0.0,
        "psi_severity": "none",
        "chi_square_stat": 0.0,
        "chi_square_pvalue": 1.0,
        "top_contributors": {},
        "action_recommended": "no_action",
    }

    if not meta["ok"] or df.empty:
        return payload

    cats = df["title"].apply(extract_strain_from_title).apply(collapse_to_categories)
    cats = cats[cats != "UNSPECIFIED"]
    cnt = _Counter(cats)
    total = sum(cnt.values())
    if total == 0:
        return payload

    current = {k: v / total for k, v in cnt.items()}
    score = psi(PUBLISHED_PREVALENCE, current)
    severity = psi_severity(score)
    stat, pval = chi_square_fit(dict(cnt), PUBLISHED_PREVALENCE)

    top = dict(list(psi_per_category(PUBLISHED_PREVALENCE, current).items())[:5])

    payload.update({
        "current_distribution": current,
        "n_sequences_typed": int(total),
        "psi_score": float(score),
        "psi_severity": severity,
        "chi_square_stat": float(stat),
        "chi_square_pvalue": float(pval),
        "top_contributors": {k: float(v) for k, v in top.items()},
        "action_recommended": (
            "retrain" if severity == "significant" else
            "monitor" if severity == "minor" else
            "no_action"
        ),
    })
    return payload


@app.get("/drift/strain/live")
def drift_strain_live(days: int = 30, force_refresh: bool = False) -> dict[str, Any]:
    """Pull HPV deposits live from NCBI and compute drift vs published baseline.

    Caches the result for 5 minutes — repeated calls within the TTL window
    return the cached payload, annotated with cache_age_seconds and
    cache_source = "cached" or "fresh".

    Pass ?force_refresh=true to bypass the cache and hit NCBI immediately.
    """
    now = datetime.now(timezone.utc)
    with _ncbi_live_lock:
        cached_at = _ncbi_live_cache["fetched_at"]
        cached_result = _ncbi_live_cache["result"]
        if (not force_refresh
                and cached_result is not None
                and cached_at is not None
                and (now - cached_at).total_seconds() < _ncbi_live_cache["ttl_seconds"]):
            age = (now - cached_at).total_seconds()
            response = dict(cached_result)
            response["cache_source"] = "cached"
            response["cache_age_seconds"] = round(age, 1)
            response["fetched_at"] = cached_at.isoformat()
            return response

    # Cache miss or forced — fetch
    logger.info("Live NCBI fetch: querying for last %d days", days)
    payload = _compute_live_drift_payload(days)
    with _ncbi_live_lock:
        _ncbi_live_cache["fetched_at"] = now
        _ncbi_live_cache["result"] = payload

    response = dict(payload)
    response["cache_source"] = "fresh"
    response["cache_age_seconds"] = 0.0
    response["fetched_at"] = now.isoformat()
    return response


# -------------------------------------------------------------------------
# Dev entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("CERVIRISK_HOST", "0.0.0.0")
    port = int(os.getenv("CERVIRISK_PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)
