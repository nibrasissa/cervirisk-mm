"""Integration tests for the CerviRisk-MM API.

These boot the FastAPI app in-process via TestClient (no real HTTP server),
exercise each endpoint, and assert response shape + invariants.

The model artifact is required — these tests skip cleanly if it doesn't
exist yet. Run `python -m src.model.train --eval split --models logreg`
to produce a quick artifact before running these tests.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.storage.paths import MODELS_DIR


# A fully-populated, plausible patient — used as the canonical happy-path input.
EXAMPLE_PATIENT = {
    "Age": 38,
    "Number of sexual partners": 4,
    "First sexual intercourse": 16,
    "Num of pregnancies": 2,
    "Smokes": 1,
    "Smokes (years)": 12,
    "Hormonal Contraceptives": 1,
    "Hormonal Contraceptives (years)": 6,
    "IUD": 0,
    "IUD (years)": 0,
    "STDs": 1,
    "STDs (number)": 1,
    "STDs:HPV": 1,
    "STDs:HIV": 0,
    "Dx:Cancer": 0,
    "Dx:CIN": 0,
    "Dx:HPV": 1,
    "Hinselmann": 0,
    "Schiller": 1,
    "Citology": 1,
    "assigned_hpv_strain": "HPV16",
    "strain_carcinogenicity": 0.95,
    "host_prs": 0.3,
    "matched_super_pop": "AMR",
}


@pytest.fixture(scope="module")
def client():
    """Boot the app once for the module, triggering the lifespan handler."""
    if not (MODELS_DIR / "cervirisk_mm_v0.1.pkl").exists():
        pytest.skip(
            "Model artifact not found. Run "
            "`python -m src.model.train --eval split --models logreg` first."
        )
    with TestClient(app) as c:
        yield c


# -------------------------------------------------------------------------
# Liveness
# -------------------------------------------------------------------------
def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "timestamp" in body


def test_root_landing(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "service" in body
    assert "docs" in body


# -------------------------------------------------------------------------
# Model info
# -------------------------------------------------------------------------
def test_model_info_shape(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    for key in ("model_name", "model_version", "model_path", "notes"):
        assert key in body
    assert body["model_name"] == "cervirisk_mm"
    assert isinstance(body["notes"], list)
    assert len(body["notes"]) >= 3


# -------------------------------------------------------------------------
# Prediction — happy path
# -------------------------------------------------------------------------
def test_predict_full_patient(client):
    r = client.post("/predict/cervical-risk", json=EXAMPLE_PATIENT)
    assert r.status_code == 200, r.text
    body = r.json()

    # Required fields present
    for key in ("risk_probability", "tier", "model_version",
                "data_status", "disclaimer"):
        assert key in body, f"missing field: {key}"

    # Probability is a valid float in [0, 1]
    assert isinstance(body["risk_probability"], (int, float))
    assert 0.0 <= body["risk_probability"] <= 1.0

    # Tier is one of three known values
    assert body["tier"] in {"low", "moderate", "high"}

    # Provenance fields
    assert body["model_version"].startswith("cervirisk_mm")
    assert body["data_status"] == "RESEARCH_PROTOTYPE"
    assert "research prototype" in body["disclaimer"].lower()


def test_predict_with_minimal_input(client):
    """An almost-empty payload should still work — the imputer handles NaN."""
    r = client.post("/predict/cervical-risk", json={"Age": 45})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_probability"] <= 1.0


def test_predict_with_empty_input(client):
    """Even {} should work — everything imputed."""
    r = client.post("/predict/cervical-risk", json={})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_probability"] <= 1.0


def test_predict_is_deterministic(client):
    """Same input must produce the same output — no randomness at inference."""
    r1 = client.post("/predict/cervical-risk", json=EXAMPLE_PATIENT)
    r2 = client.post("/predict/cervical-risk", json=EXAMPLE_PATIENT)
    assert r1.json()["risk_probability"] == r2.json()["risk_probability"]


# -------------------------------------------------------------------------
# Prediction — validation
# -------------------------------------------------------------------------
def test_predict_rejects_invalid_age(client):
    """Age outside [10, 110] must 422."""
    r = client.post("/predict/cervical-risk", json={"Age": 500})
    assert r.status_code == 422


def test_predict_rejects_invalid_binary(client):
    """Binary flag outside [0, 1] must 422."""
    r = client.post("/predict/cervical-risk", json={"Smokes": 5})
    assert r.status_code == 422


def test_predict_ignores_unknown_fields(client):
    """Extra fields should not crash the request."""
    payload = dict(EXAMPLE_PATIENT)
    payload["random_extra_field"] = "anything"
    r = client.post("/predict/cervical-risk", json=payload)
    assert r.status_code == 200


# -------------------------------------------------------------------------
# OpenAPI docs are reachable
# -------------------------------------------------------------------------
def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["info"]["title"] == "CerviRisk-MM"
    paths = spec["paths"]
    assert "/health" in paths
    assert "/model/info" in paths
    assert "/predict/cervical-risk" in paths
