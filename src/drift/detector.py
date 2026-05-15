"""Drift detection for CerviRisk-MM.

Three statistical tests, each picking up a different kind of distribution shift:

  - psi()              : Population Stability Index. For categorical features
                          (assigned_hpv_strain, matched_super_pop) or binned
                          numeric features. Industry-standard thresholds:
                            PSI < 0.10  → no drift
                            0.10 ≤ PSI < 0.20  → minor drift, monitor
                            PSI ≥ 0.20  → significant drift, investigate

  - ks_two_sample()    : Kolmogorov-Smirnov test. For continuous features
                          (Age, host_prs). Returns the test statistic D and
                          a p-value. Threshold:
                            D ≥ 0.10 OR p < 0.05  → drift

  - chi_square_fit()   : Chi-square goodness of fit. For comparing an
                          observed categorical distribution against an
                          expected proportion vector (e.g. NCBI strain mix
                          vs. published de Sanjosé 2010 prior). Returns the
                          test statistic and a p-value. Threshold:
                            p < 0.05  → drift

Each function is independent and pure: same inputs, same outputs, no side
effects. The Detector class composes them into a per-feature drift report.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# Thresholds — documented and configurable, not magic numbers
PSI_MINOR = 0.10
PSI_SIGNIFICANT = 0.20
KS_STATISTIC_THRESHOLD = 0.10
P_VALUE_THRESHOLD = 0.05

_EPSILON = 1e-6   # smoothing for zero probabilities


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------
def psi(
    reference_proportions: dict[str, float],
    current_proportions: dict[str, float],
) -> float:
    """Population Stability Index between two categorical distributions.

    PSI = Σ_i (current_p_i − reference_p_i) × ln(current_p_i / reference_p_i)

    Both inputs are dicts mapping category → proportion (∈ [0, 1]).
    The category sets need not match; missing categories are smoothed with ε.

    Returns a non-negative float. Larger = more drift.
    """
    categories = set(reference_proportions) | set(current_proportions)
    score = 0.0
    for cat in categories:
        ref = max(reference_proportions.get(cat, 0.0), _EPSILON)
        cur = max(current_proportions.get(cat, 0.0), _EPSILON)
        score += (cur - ref) * math.log(cur / ref)
    return float(score)


def psi_per_category(
    reference_proportions: dict[str, float],
    current_proportions: dict[str, float],
) -> dict[str, float]:
    """Per-category PSI contribution — useful for explaining which categories drove drift."""
    categories = set(reference_proportions) | set(current_proportions)
    contrib: dict[str, float] = {}
    for cat in categories:
        ref = max(reference_proportions.get(cat, 0.0), _EPSILON)
        cur = max(current_proportions.get(cat, 0.0), _EPSILON)
        contrib[cat] = float((cur - ref) * math.log(cur / ref))
    return dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1])))


def ks_two_sample(
    reference_values: list[float] | np.ndarray,
    current_values: list[float] | np.ndarray,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test on continuous values.

    Returns (statistic D, p-value). D is the largest vertical distance
    between the two empirical CDFs. Under the null (same distribution),
    D is small and the p-value is high.
    """
    ref = np.asarray(reference_values, dtype=float)
    cur = np.asarray(current_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, 1.0
    result = stats.ks_2samp(ref, cur, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def chi_square_fit(
    observed_counts: dict[str, int],
    expected_proportions: dict[str, float],
) -> tuple[float, float]:
    """Chi-square goodness of fit between observed counts and expected proportions.

    Uses the union of categories from both inputs. Categories absent from
    expected_proportions get an ε prior so the statistic is defined.

    Returns (chi-square statistic, p-value).
    """
    categories = sorted(set(observed_counts) | set(expected_proportions))
    obs = np.array([observed_counts.get(c, 0) for c in categories], dtype=float)
    exp_p = np.array([max(expected_proportions.get(c, 0.0), _EPSILON)
                       for c in categories], dtype=float)
    exp_p = exp_p / exp_p.sum()
    exp_counts = exp_p * obs.sum()

    if obs.sum() == 0 or exp_counts.sum() == 0:
        return 0.0, 1.0

    stat = float(((obs - exp_counts) ** 2 / exp_counts).sum())
    dof = max(len(categories) - 1, 1)
    pvalue = float(1.0 - stats.chi2.cdf(stat, dof))
    return stat, pvalue


# ---------------------------------------------------------------------------
# Helpers — interpret a metric value as a drift severity
# ---------------------------------------------------------------------------
def psi_severity(score: float) -> str:
    if score < PSI_MINOR:
        return "none"
    if score < PSI_SIGNIFICANT:
        return "minor"
    return "significant"


def ks_drift_detected(statistic: float, pvalue: float) -> bool:
    return statistic >= KS_STATISTIC_THRESHOLD or pvalue < P_VALUE_THRESHOLD


def chi_square_drift_detected(pvalue: float) -> bool:
    return pvalue < P_VALUE_THRESHOLD


# ---------------------------------------------------------------------------
# Per-feature report container
# ---------------------------------------------------------------------------
@dataclass
class FeatureDriftReport:
    feature: str
    method: str               # "psi" | "ks" | "chi_square"
    score: float
    pvalue: float | None
    severity: str             # "none" | "minor" | "significant"
    drift_detected: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "method": self.method,
            "score": round(self.score, 4),
            "pvalue": (round(self.pvalue, 4) if self.pvalue is not None else None),
            "severity": self.severity,
            "drift_detected": self.drift_detected,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Detector — composes the three pure functions into a per-feature scan
# ---------------------------------------------------------------------------
class DriftDetector:
    """Run all three drift tests against a fixed baseline.

    The baseline is captured once on training data (see src/drift/baseline.py);
    the detector is invoked at prediction time against batches of new data.
    """

    def __init__(self, baseline: dict[str, Any]) -> None:
        self.baseline = baseline

    # --- Categorical / strain comparison -------------------------------------
    def check_categorical(
        self, feature: str, current_values: list[str]
    ) -> FeatureDriftReport:
        """PSI against the saved baseline proportions for this feature."""
        ref_info = self.baseline.get("categorical_features", {}).get(feature)
        if not ref_info:
            return FeatureDriftReport(feature, "psi", 0.0, None, "none", False,
                                       {"note": "feature not in baseline"})

        # Build the current proportions from the incoming values.
        total = max(len(current_values), 1)
        from collections import Counter
        cnt = Counter(str(v) for v in current_values if v is not None)
        current_props = {k: v / total for k, v in cnt.items()}

        score = psi(ref_info["proportions"], current_props)
        severity = psi_severity(score)
        contributions = psi_per_category(ref_info["proportions"], current_props)
        # Keep the top contributors only (so the JSON doesn't bloat)
        top = dict(list(contributions.items())[:5])
        return FeatureDriftReport(
            feature, "psi", score, None, severity, severity != "none",
            {"top_contributors": top,
             "n_reference": ref_info.get("n", None),
             "n_current": total},
        )

    # --- Continuous comparison -----------------------------------------------
    def check_numeric(
        self, feature: str, current_values: list[float]
    ) -> FeatureDriftReport:
        """KS two-sample test against the baseline's stored values."""
        ref_info = self.baseline.get("numeric_features", {}).get(feature)
        if not ref_info or "values_sample" not in ref_info:
            return FeatureDriftReport(feature, "ks", 0.0, 1.0, "none", False,
                                       {"note": "feature not in baseline"})

        stat, pval = ks_two_sample(ref_info["values_sample"], current_values)
        detected = ks_drift_detected(stat, pval)
        severity = "significant" if detected else "none"
        return FeatureDriftReport(
            feature, "ks", stat, pval, severity, detected,
            {"n_reference": len(ref_info["values_sample"]),
             "n_current": len(current_values),
             "current_mean": (float(np.nanmean(current_values))
                              if len(current_values) else None),
             "current_std": (float(np.nanstd(current_values))
                              if len(current_values) else None),
             "reference_mean": ref_info.get("mean"),
             "reference_std": ref_info.get("std")},
        )

    # --- Vs published prior (de Sanjose 2010 for HPV strains) ----------------
    def check_against_prior(
        self,
        feature: str,
        observed_counts: dict[str, int],
        expected_proportions: dict[str, float],
    ) -> FeatureDriftReport:
        """Chi-square fit of observed counts to an expected categorical prior.

        Used for NCBI strain composition vs. the published de Sanjose 2010
        prevalence priors — a different question from 'does this batch
        differ from training data?'.
        """
        stat, pval = chi_square_fit(observed_counts, expected_proportions)
        detected = chi_square_drift_detected(pval)
        severity = "significant" if detected else "none"
        return FeatureDriftReport(
            feature, "chi_square", stat, pval, severity, detected,
            {"observed_counts": observed_counts,
             "expected_proportions": expected_proportions},
        )

    # --- Whole-batch scan -----------------------------------------------------
    def scan(self, current_df) -> dict[str, Any]:
        """Run all configured checks on a DataFrame and return a full report."""
        import pandas as pd

        if not isinstance(current_df, pd.DataFrame):
            raise TypeError("scan() expects a pandas DataFrame")

        reports: list[FeatureDriftReport] = []
        for col, info in self.baseline.get("numeric_features", {}).items():
            if col in current_df.columns:
                vals = current_df[col].dropna().astype(float).tolist()
                if vals:
                    reports.append(self.check_numeric(col, vals))

        for col, info in self.baseline.get("categorical_features", {}).items():
            if col in current_df.columns:
                vals = current_df[col].dropna().astype(str).tolist()
                if vals:
                    reports.append(self.check_categorical(col, vals))

        any_drift = any(r.drift_detected for r in reports)
        significant = [r for r in reports if r.severity == "significant"]

        return {
            "drift_detected": any_drift,
            "n_features_checked": len(reports),
            "n_features_with_drift": sum(1 for r in reports if r.drift_detected),
            "severity_summary": {
                "significant": len(significant),
                "minor": sum(1 for r in reports if r.severity == "minor"),
                "none": sum(1 for r in reports if r.severity == "none"),
            },
            "action_recommended": (
                "retrain" if any_drift and significant else
                "monitor" if any_drift else
                "no_action"
            ),
            "by_feature": [r.to_dict() for r in reports],
        }
