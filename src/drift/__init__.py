"""Drift detection for CerviRisk-MM.

Public API:
    from src.drift import DriftDetector, capture_baseline, save_baseline, load_baseline
    from src.drift import psi, ks_two_sample, chi_square_fit
"""
from src.drift.baseline import (
    BASELINE_PATH,
    capture_baseline,
    load_baseline,
    save_baseline,
)
from src.drift.detector import (
    DriftDetector,
    FeatureDriftReport,
    chi_square_fit,
    chi_square_drift_detected,
    ks_drift_detected,
    ks_two_sample,
    psi,
    psi_per_category,
    psi_severity,
)

__all__ = [
    "DriftDetector",
    "FeatureDriftReport",
    "BASELINE_PATH",
    "capture_baseline",
    "load_baseline",
    "save_baseline",
    "psi",
    "psi_per_category",
    "psi_severity",
    "ks_two_sample",
    "ks_drift_detected",
    "chi_square_fit",
    "chi_square_drift_detected",
]
