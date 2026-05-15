"""Per-prediction explanation and multi-modal audit for CerviRisk-MM.

Two responsibilities:

1. `explain_prediction(pipeline, patient_df)` — runs SHAP TreeExplainer on
   the trained pipeline and returns the top-K feature contributions for a
   single patient, ranked by absolute impact on the predicted probability.
   Falls back gracefully if SHAP isn't installed or the classifier isn't
   tree-based.

2. `build_audit(patient_dict, baseline=None)` — assembles the multi-modal
   context the model used: host genetics (PRS + ancestry), HPV viral
   (strain + carcinogenicity), and clinical aggregate (screening test
   count + lifestyle risk indicators). This is the "why did the model
   see this patient as multi-modal" answer.

Both functions are designed to be called from the FastAPI /predict
endpoint and embedded in the response JSON.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display-name lookup — makes SHAP output human-readable
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "Age": "Age",
    "Number of sexual partners": "Number of sexual partners",
    "First sexual intercourse": "Age at first intercourse",
    "Num of pregnancies": "Pregnancies",
    "Smokes": "Smoking status",
    "Smokes (years)": "Years smoking",
    "Hormonal Contraceptives": "Hormonal contraceptives",
    "Hormonal Contraceptives (years)": "Years on hormonal contraceptives",
    "IUD": "IUD",
    "IUD (years)": "Years with IUD",
    "STDs": "Any STDs",
    "STDs (number)": "Number of STDs",
    "STDs:HPV": "Prior HPV STD",
    "STDs:HIV": "Prior HIV",
    "Dx:Cancer": "Prior cancer diagnosis",
    "Dx:CIN": "Prior CIN diagnosis",
    "Dx:HPV": "Prior HPV diagnosis",
    "Hinselmann": "Hinselmann test (acetowhite)",
    "Schiller": "Schiller test (iodine staining)",
    "Citology": "Cytology test (Pap)",
    "host_prs": "Polygenic risk score",
    "strain_carcinogenicity": "HPV strain carcinogenicity",
    "assigned_hpv_strain": "HPV strain",
    "matched_super_pop": "Ancestry group",
}


def _clean_feature_name(raw: str) -> str:
    """Convert 'num__Age' -> 'Age', 'cat__assigned_hpv_strain_HPV16' -> 'HPV strain: HPV16'."""
    name = raw
    if "__" in name:
        name = name.split("__", 1)[1]
    for prefix in ("assigned_hpv_strain_", "matched_super_pop_"):
        if name.startswith(prefix):
            base_key = prefix.rstrip("_")
            value = name[len(prefix):]
            label = FEATURE_DISPLAY_NAMES.get(base_key, base_key)
            return f"{label}: {value}"
    return FEATURE_DISPLAY_NAMES.get(name, name)


# ---------------------------------------------------------------------------
# Multi-modal context audit
# ---------------------------------------------------------------------------
def _carcinogenicity_for(strain: str | None) -> tuple[float | None, str]:
    """Map a strain code to its IARC carcinogenicity weight and group label."""
    try:
        from src.features.strain import CARCINOGENICITY
    except Exception:
        return None, "unknown"

    if not strain or strain not in CARCINOGENICITY:
        return None, "unknown"

    w = CARCINOGENICITY[strain]
    if w >= 0.8:
        return w, "Group 1 — carcinogenic"
    if w >= 0.5:
        return w, "Group 2A/2B — probably carcinogenic"
    if w >= 0.2:
        return w, "Possibly carcinogenic"
    return w, "Low-risk / non-carcinogenic"


def _published_prevalence(strain: str | None) -> float | None:
    try:
        from src.features.strain import PUBLISHED_PREVALENCE
        return PUBLISHED_PREVALENCE.get(strain) if strain else None
    except Exception:
        return None


def _prs_percentile(prs: float | None, baseline: dict | None) -> int | None:
    """Compute the percentile of this patient's PRS against the training distribution."""
    if prs is None or baseline is None:
        return None
    try:
        sample = baseline.get("numeric_features", {}).get("host_prs", {}).get("values_sample")
        if not sample:
            return None
        arr = np.asarray(sample, dtype=float)
        return int(round(100 * (arr <= prs).mean()))
    except Exception:
        return None


def _clinical_aggregate(p: dict) -> dict:
    """Summarize the clinical evidence (independent of the model)."""
    screening = sum(
        int(bool(p.get(k, 0))) for k in ("Hinselmann", "Schiller", "Citology")
    )
    risk_flags = 0
    if p.get("Smokes", 0):
        risk_flags += 1
    fsi = p.get("First sexual intercourse")
    if fsi is not None and fsi < 16:
        risk_flags += 1
    nsp = p.get("Number of sexual partners")
    if nsp is not None and nsp >= 5:
        risk_flags += 1
    if p.get("STDs", 0) or p.get("STDs:HPV", 0):
        risk_flags += 1
    return {
        "screening_tests_positive": screening,
        "screening_tests_total": 3,
        "high_risk_factors_count": risk_flags,
        "high_risk_factors_max": 4,
    }


def build_audit(patient: dict, baseline: dict | None = None) -> dict:
    """Construct the multi-modal context audit for a single patient.

    All sections degrade gracefully when inputs are missing — fields will
    be `None` and the UI can hide them rather than show an error.
    """
    strain = patient.get("assigned_hpv_strain")
    carcin, iarc = _carcinogenicity_for(strain)
    prs = patient.get("host_prs")
    prs_percentile = _prs_percentile(prs, baseline)

    host = {
        "polygenic_risk_score": (round(prs, 3) if prs is not None else None),
        "percentile_in_training_distribution": prs_percentile,
        "matched_super_pop": patient.get("matched_super_pop"),
        "interpretation": (
            "elevated" if prs_percentile is not None and prs_percentile >= 75 else
            "average" if prs_percentile is not None and prs_percentile >= 25 else
            "lower" if prs_percentile is not None else
            "not_provided"
        ),
    }

    viral = {
        "assigned_strain": strain,
        "carcinogenicity": (round(carcin, 2) if carcin is not None else None),
        "iarc_classification": iarc,
        "published_prevalence": _published_prevalence(strain),
    }

    return {
        "host_genetics": host,
        "hpv_viral": viral,
        "clinical": _clinical_aggregate(patient),
    }


# ---------------------------------------------------------------------------
# SHAP feature contributions
# ---------------------------------------------------------------------------
def _compute_xgboost_shap_via_booster(inner, X_t, feature_names):
    """Compute SHAP values using XGBoost's built-in pred_contribs.

    This avoids the SHAP library entirely — it uses XGBoost's own
    TreeSHAP implementation. More reliable across XGBoost+SHAP version
    combinations and faster.

    Returns a 1-D array of contributions for the first row of X_t.
    """
    import xgboost as xgb
    booster = inner.get_booster()
    # Try to set feature names so DMatrix matches the booster's expectations
    try:
        booster_feature_names = booster.feature_names
        if booster_feature_names:
            # Use booster's own names to avoid mismatch
            dmat = xgb.DMatrix(X_t, feature_names=booster_feature_names)
        else:
            dmat = xgb.DMatrix(X_t, feature_names=feature_names)
    except Exception:
        dmat = xgb.DMatrix(X_t)
    # contribs shape: (n_samples, n_features + 1). Last column is bias.
    contribs = booster.predict(dmat, pred_contribs=True)
    return contribs[0, :-1]


def explain_prediction(
    pipeline: Any,
    patient_df: pd.DataFrame,
    top_k: int = 10,
) -> dict | None:
    """Compute top-K feature contributions for a single-patient prediction.

    For XGBoost classifiers: uses XGBoost's built-in pred_contribs (TreeSHAP)
    For other tree-based classifiers: falls back to SHAP TreeExplainer

    Returns a dict with `error` field if everything fails, so the UI can
    surface the actual reason rather than say "not available".
    """
    if not hasattr(pipeline, "named_steps"):
        return {
            "method": "unavailable",
            "error": "Pipeline shape unexpected (no named_steps attribute)",
            "top_contributors": [],
        }

    pre = pipeline.named_steps.get("pre") or pipeline.named_steps.get("preprocessor")
    clf = pipeline.named_steps.get("clf") or pipeline.named_steps.get("classifier")
    if pre is None or clf is None:
        return {
            "method": "unavailable",
            "error": f"Could not find preprocessor/classifier. "
                     f"Available steps: {list(pipeline.named_steps.keys())}",
            "top_contributors": [],
        }

    # For CalibratedClassifierCV, unwrap to the underlying estimator
    inner = clf
    inner_type = type(clf).__name__
    if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
        inner = clf.calibrated_classifiers_[0].estimator
        inner_type = type(inner).__name__ + " (unwrapped from CalibratedClassifierCV)"

    # Detect XGBoost — its built-in pred_contribs is the most reliable path
    is_xgboost = (
        type(inner).__module__.startswith("xgboost")
        or hasattr(inner, "get_booster")
    )

    try:
        X_t = pre.transform(patient_df)
        try:
            feature_names = list(pre.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(X_t.shape[1])]

        method_used: str
        errors: list[str] = []

        if is_xgboost:
            # Path 1: XGBoost built-in pred_contribs (TreeSHAP, no SHAP lib needed)
            try:
                shap_row = _compute_xgboost_shap_via_booster(
                    inner, X_t, feature_names
                )
                method_used = "XGBoost pred_contribs (built-in TreeSHAP)"
            except Exception as e:
                errors.append(f"xgboost.pred_contribs: {type(e).__name__}: {e}")
                shap_row = None
        else:
            shap_row = None

        # Path 2: fallback to SHAP TreeExplainer (for non-XGBoost or if path 1 failed)
        if shap_row is None:
            try:
                import shap
                explainer = shap.TreeExplainer(inner)
                sv = explainer.shap_values(X_t)
                if isinstance(sv, list):
                    sv = sv[1]
                sv = np.asarray(sv)
                if sv.ndim == 2:
                    sv = sv[0]
                shap_row = sv
                method_used = "SHAP TreeExplainer"
            except ImportError:
                errors.append("SHAP not installed (pip install shap)")
            except Exception as e:
                errors.append(f"SHAP TreeExplainer: {type(e).__name__}: {e}")

        if shap_row is None:
            return {
                "method": "unavailable",
                "error": " | ".join(errors) or "no working SHAP path",
                "classifier_type": inner_type,
                "top_contributors": [],
            }

        # Build ranked output
        x_row = X_t[0] if hasattr(X_t, "__getitem__") else np.asarray(X_t).ravel()
        ranked = []
        for raw_name, value, contribution in zip(feature_names, x_row, shap_row):
            ranked.append({
                "feature": raw_name,
                "display_name": _clean_feature_name(raw_name),
                "value": float(value),
                "contribution": float(contribution),
                "direction": ("increases_risk" if contribution > 0
                               else "decreases_risk" if contribution < 0
                               else "neutral"),
            })

        ranked.sort(key=lambda r: -abs(r["contribution"]))
        ranked = ranked[:top_k]

        return {
            "method": method_used,
            "classifier_type": inner_type,
            "n_features_in_model": len(feature_names),
            "top_contributors": ranked,
        }

    except Exception as e:
        logger.warning("Explanation failed unexpectedly: %s: %s",
                       type(e).__name__, e)
        return {
            "method": "unavailable",
            "error": f"unexpected: {type(e).__name__}: {str(e)[:300]}",
            "classifier_type": inner_type,
            "top_contributors": [],
        }
