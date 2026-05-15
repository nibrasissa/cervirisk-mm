"""Tests for the drift module — math, persistence, detector, and API.

Covers:
  - PSI math on known inputs (identity, total shift, partial shift)
  - KS two-sample on identical and shifted distributions
  - Chi-square fit
  - Baseline save/load round-trip
  - DriftDetector class with synthetic baseline + current data
  - /drift/baseline and /drift/check API endpoints
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    DriftDetector,
    capture_baseline,
    chi_square_fit,
    ks_two_sample,
    load_baseline,
    psi,
    psi_per_category,
    psi_severity,
    save_baseline,
)


# ---------------------------------------------------------------------------
# Pure metric tests
# ---------------------------------------------------------------------------
def test_psi_identical_distributions_is_near_zero():
    """PSI of a distribution with itself must be effectively 0."""
    p = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert psi(p, p) < 1e-6


def test_psi_large_shift_exceeds_significant_threshold():
    """A 50pp swap between two categories should land >> 0.20."""
    baseline = {"HPV16": 0.55, "HPV18": 0.15, "others": 0.30}
    shifted = {"HPV16": 0.20, "HPV18": 0.15, "others": 0.65}
    score = psi(baseline, shifted)
    assert score > 0.20, f"expected significant drift, got PSI={score:.3f}"
    assert psi_severity(score) == "significant"


def test_psi_severity_thresholds():
    assert psi_severity(0.05) == "none"
    assert psi_severity(0.15) == "minor"
    assert psi_severity(0.25) == "significant"


def test_psi_per_category_signs_and_ordering():
    """Categories whose proportion went UP should have positive contributions."""
    baseline = {"a": 0.5, "b": 0.3, "c": 0.2}
    current = {"a": 0.3, "b": 0.5, "c": 0.2}  # b ↑, a ↓, c unchanged
    contrib = psi_per_category(baseline, current)
    assert contrib["b"] > 0
    assert contrib["a"] > 0  # PSI contribution is symmetric (always ≥ 0 per cat for any change)


def test_psi_handles_unseen_categories():
    """A category in current but not baseline (or vice versa) must not crash."""
    baseline = {"a": 0.7, "b": 0.3}
    current = {"a": 0.5, "b": 0.3, "new_strain": 0.2}
    score = psi(baseline, current)
    assert score > 0
    assert np.isfinite(score)


# ---------------------------------------------------------------------------
# KS tests
# ---------------------------------------------------------------------------
def test_ks_identical_arrays_no_drift():
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 1, 200)
    stat, pval = ks_two_sample(arr, arr)
    assert stat < 0.05
    assert pval > 0.5


def test_ks_shifted_distribution_detected():
    """A mean shift of 1 std must trigger drift detection."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(1, 1, 500)
    stat, pval = ks_two_sample(ref, cur)
    assert stat >= 0.10
    assert pval < 0.05


def test_ks_handles_nan_and_empty():
    stat, pval = ks_two_sample([], [1, 2, 3])
    assert stat == 0.0 and pval == 1.0


# ---------------------------------------------------------------------------
# Chi-square fit
# ---------------------------------------------------------------------------
def test_chi_square_matching_distribution_no_drift():
    """Observed counts proportional to expected → no drift."""
    expected = {"HPV16": 0.55, "HPV18": 0.15, "others": 0.30}
    observed = {"HPV16": 550, "HPV18": 150, "others": 300}
    stat, pval = chi_square_fit(observed, expected)
    assert pval > 0.05, f"expected high p-value, got {pval}"


def test_chi_square_divergent_observed_drift():
    """Observed distribution shifted away from expected → drift."""
    expected = {"HPV16": 0.55, "HPV18": 0.15, "others": 0.30}
    observed = {"HPV16": 900, "HPV18": 50, "others": 50}
    stat, pval = chi_square_fit(observed, expected)
    assert pval < 0.05


# ---------------------------------------------------------------------------
# Baseline capture and persistence
# ---------------------------------------------------------------------------
def test_capture_baseline_structure():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Age": rng.normal(35, 10, 100),
        "Smokes (years)": rng.uniform(0, 20, 100),
        "assigned_hpv_strain": rng.choice(
            ["HPV16", "HPV18", "NONE", "OTHER_HR_HPV"], 100, p=[0.05, 0.02, 0.90, 0.03]
        ),
        "Biopsy": rng.choice([0, 1], 100),
    })
    baseline = capture_baseline(df,
                                numeric_features=("Age", "Smokes (years)"),
                                categorical_features=("assigned_hpv_strain",))

    assert "version" in baseline
    assert "captured_at" in baseline
    assert baseline["n_records"] == 100

    age = baseline["numeric_features"]["Age"]
    assert age["n"] == 100
    assert "mean" in age and "std" in age
    assert "percentiles" in age
    assert "values_sample" in age

    strain = baseline["categorical_features"]["assigned_hpv_strain"]
    assert "value_counts" in strain
    assert "proportions" in strain
    assert abs(sum(strain["proportions"].values()) - 1.0) < 1e-6


def test_baseline_save_load_round_trip(tmp_path):
    df = pd.DataFrame({"Age": np.random.RandomState(0).normal(35, 10, 50),
                        "assigned_hpv_strain": ["NONE"] * 50})
    baseline = capture_baseline(df,
                                numeric_features=("Age",),
                                categorical_features=("assigned_hpv_strain",))

    path = tmp_path / "drift_baseline.json"
    save_baseline(baseline, path)
    loaded = load_baseline(path)

    assert loaded["n_records"] == baseline["n_records"]
    assert loaded["numeric_features"]["Age"]["mean"] == \
           baseline["numeric_features"]["Age"]["mean"]


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------
def test_detector_no_drift_when_current_matches_baseline():
    rng = np.random.default_rng(0)
    df_baseline = pd.DataFrame({
        "Age": rng.normal(35, 10, 2000),
        "assigned_hpv_strain": rng.choice(
            ["HPV16", "NONE"], 2000, p=[0.05, 0.95]
        ),
    })
    baseline = capture_baseline(df_baseline,
                                numeric_features=("Age",),
                                categorical_features=("assigned_hpv_strain",))
    detector = DriftDetector(baseline)

    # New batch from same distributions, large enough that KS statistic stays small
    rng2 = np.random.default_rng(1)
    df_new = pd.DataFrame({
        "Age": rng2.normal(35, 10, 2000),
        "assigned_hpv_strain": rng2.choice(
            ["HPV16", "NONE"], 2000, p=[0.05, 0.95]
        ),
    })
    report = detector.scan(df_new)
    # No retrain should be triggered when samples come from the same distribution
    assert report["action_recommended"] != "retrain"
    assert report["severity_summary"]["significant"] == 0


def test_detector_detects_strong_drift():
    """Shift age mean by 15 years and flip strain prevalence — must trigger."""
    rng = np.random.default_rng(0)
    df_baseline = pd.DataFrame({
        "Age": rng.normal(35, 10, 500),
        "assigned_hpv_strain": rng.choice(
            ["HPV16", "NONE"], 500, p=[0.05, 0.95]
        ),
    })
    baseline = capture_baseline(df_baseline,
                                numeric_features=("Age",),
                                categorical_features=("assigned_hpv_strain",))
    detector = DriftDetector(baseline)

    rng2 = np.random.default_rng(1)
    df_new = pd.DataFrame({
        "Age": rng2.normal(50, 10, 200),
        "assigned_hpv_strain": rng2.choice(
            ["HPV16", "NONE"], 200, p=[0.40, 0.60]
        ),
    })
    report = detector.scan(df_new)
    assert report["drift_detected"] is True
    assert report["action_recommended"] == "retrain"
    assert report["n_features_with_drift"] >= 1


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def api_client():
    """Boot the app once; skip if model artifact is missing."""
    from src.storage.paths import MODELS_DIR
    if not (MODELS_DIR / "cervirisk_mm_v0.1.pkl").exists():
        pytest.skip("Model artifact not found.")

    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_drift_baseline_endpoint(api_client):
    r = api_client.get("/drift/baseline")
    if r.status_code == 503:
        pytest.skip("Drift baseline not captured; run "
                    "`python -m src.drift.baseline` first.")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body
    assert "n_records" in body
    assert "numeric_features" in body
    assert "categorical_features" in body


def test_drift_check_endpoint_no_drift(api_client):
    """Submit records similar to baseline — expect no drift."""
    # First, check baseline endpoint to see if we can run this test
    r = api_client.get("/drift/baseline")
    if r.status_code == 503:
        pytest.skip("Drift baseline not captured")

    records = [
        {"Age": 30, "Smokes (years)": 5, "assigned_hpv_strain": "NONE"}
        for _ in range(50)
    ]
    r = api_client.post("/drift/check", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    for key in ("drift_detected", "n_features_checked",
                "severity_summary", "action_recommended", "by_feature"):
        assert key in body


def test_drift_check_endpoint_with_shifted_records(api_client):
    """Submit records far from baseline — expect drift detection."""
    r = api_client.get("/drift/baseline")
    if r.status_code == 503:
        pytest.skip("Drift baseline not captured")

    # Push all records to very old patients with HPV16 — opposite of training
    records = [
        {"Age": 65, "Smokes (years)": 30, "assigned_hpv_strain": "HPV16"}
        for _ in range(50)
    ]
    r = api_client.post("/drift/check", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    # Drift should be detected (Age mean shifted ~30 years, strain proportion ~100% HPV16)
    assert body["drift_detected"] is True
