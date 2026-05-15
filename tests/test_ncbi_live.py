"""Tests for the live NCBI ingestion module and the /drift/strain/live endpoint.

These tests mock the NCBI Entrez calls so they run offline. They verify:
  - The query builder produces a valid date-bounded search
  - fetch_recent_deposits returns a DataFrame + meta dict
  - Errors from NCBI are caught and surfaced via meta, not raised
  - Empty NCBI responses produce an empty DataFrame
  - The /drift/strain/live endpoint caches results for 5 minutes
  - force_refresh=true bypasses the cache
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.ingestion import ncbi_live


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------
def test_query_builder_includes_pdat_window():
    q = ncbi_live._build_query(days=30)
    assert '"last 30 days"[PDAT]' in q
    assert "papillomavirus" in q.lower()


def test_fetch_recent_deposits_handles_empty_response():
    """If NCBI returns no IDs, fetch returns empty DataFrame + meta.ok=True."""
    with patch.object(ncbi_live, "_do_search", return_value=[]):
        df, meta = ncbi_live.fetch_recent_deposits(days=30)
    assert df.empty
    assert meta["ok"] is True
    assert meta["n_records"] == 0


def test_fetch_recent_deposits_returns_records():
    """A successful fetch parses summaries into the expected schema."""
    fake_ids = ["123", "456"]
    fake_summaries = [
        {"AccessionVersion": "KX000001.1", "Title": "HPV16 L1 gene, complete"},
        {"AccessionVersion": "KX000002.1", "Title": "HPV18 isolate XYZ"},
    ]
    with patch.object(ncbi_live, "_do_search", return_value=fake_ids), \
         patch.object(ncbi_live, "_do_summary", return_value=fake_summaries):
        df, meta = ncbi_live.fetch_recent_deposits(days=30)

    assert len(df) == 2
    assert list(df.columns) == ["accession", "title", "organism"]
    assert df.iloc[0]["accession"] == "KX000001.1"
    assert "HPV16" in df.iloc[0]["title"]
    assert meta["ok"] is True
    assert meta["n_records"] == 2


def test_fetch_recent_deposits_handles_persistent_failure():
    """If NCBI errors twice (initial + retry), meta.ok = False, no exception."""
    def boom(*a, **kw):
        raise RuntimeError("NCBI 503 Service Unavailable")

    with patch.object(ncbi_live, "_do_search", side_effect=boom):
        df, meta = ncbi_live.fetch_recent_deposits(days=30)

    assert df.empty
    assert meta["ok"] is False
    assert "503" in meta["error"]


def test_fetch_recent_deposits_retries_on_transient_failure():
    """First call raises, second call succeeds → we still get records."""
    call_count = {"n": 0}

    def flaky(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient")
        return ["123"]

    with patch.object(ncbi_live, "_do_search", side_effect=flaky), \
         patch.object(ncbi_live, "_do_summary",
                      return_value=[{"AccessionVersion": "X1",
                                     "Title": "HPV16 sequence"}]):
        df, meta = ncbi_live.fetch_recent_deposits(days=30)

    assert call_count["n"] == 2
    assert meta["ok"] is True
    assert len(df) == 1


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client():
    """Boot the app; skip if model artifact missing."""
    from src.storage.paths import MODELS_DIR
    if not (MODELS_DIR / "cervirisk_mm_v0.1.pkl").exists():
        pytest.skip("Model artifact not found.")
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def _reset_live_cache():
    """Clear the live-NCBI cache so each test starts fresh."""
    from src.api.main import _ncbi_live_cache
    _ncbi_live_cache["fetched_at"] = None
    _ncbi_live_cache["result"] = None


def _fake_ncbi_payload():
    """Reasonable HPV16-dominant strain distribution."""
    return [
        {"AccessionVersion": f"X{i:05d}", "Title": f"HPV16 isolate {i}"}
        for i in range(50)
    ] + [
        {"AccessionVersion": f"Y{i:05d}", "Title": f"HPV18 isolate {i}"}
        for i in range(5)
    ]


def test_drift_strain_live_endpoint_returns_payload(api_client):
    _reset_live_cache()
    with patch.object(ncbi_live, "_do_search",
                      return_value=[str(i) for i in range(55)]), \
         patch.object(ncbi_live, "_do_summary",
                      return_value=_fake_ncbi_payload()):
        r = api_client.get("/drift/strain/live")
    assert r.status_code == 200
    body = r.json()

    for key in ("psi_score", "psi_severity", "current_distribution",
                "baseline_distribution", "n_sequences_typed",
                "cache_source", "cache_age_seconds", "action_recommended"):
        assert key in body, f"missing field: {key}"

    assert body["cache_source"] == "fresh"
    assert body["cache_age_seconds"] == 0.0
    assert body["n_sequences_typed"] > 0
    # HPV16-dominant deposits should produce significant PSI vs baseline
    assert body["psi_score"] > 0.20


def test_drift_strain_live_uses_cache_on_second_call(api_client):
    _reset_live_cache()
    with patch.object(ncbi_live, "_do_search",
                      return_value=[str(i) for i in range(10)]) as mock_search, \
         patch.object(ncbi_live, "_do_summary",
                      return_value=_fake_ncbi_payload()[:10]):
        r1 = api_client.get("/drift/strain/live")
        r2 = api_client.get("/drift/strain/live")

    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["cache_source"] == "fresh"
    assert r2.json()["cache_source"] == "cached"
    # NCBI was hit exactly once (second call served from cache)
    assert mock_search.call_count == 1


def test_drift_strain_live_force_refresh_bypasses_cache(api_client):
    _reset_live_cache()
    with patch.object(ncbi_live, "_do_search",
                      return_value=[str(i) for i in range(10)]) as mock_search, \
         patch.object(ncbi_live, "_do_summary",
                      return_value=_fake_ncbi_payload()[:10]):
        api_client.get("/drift/strain/live")
        r2 = api_client.get("/drift/strain/live?force_refresh=true")

    assert r2.json()["cache_source"] == "fresh"
    assert mock_search.call_count == 2


def test_drift_strain_live_handles_ncbi_failure(api_client):
    """If NCBI errors, endpoint returns 200 with meta.ok=False — no 500."""
    _reset_live_cache()
    with patch.object(ncbi_live, "_do_search",
                      side_effect=RuntimeError("NCBI 503")):
        r = api_client.get("/drift/strain/live")

    assert r.status_code == 200
    body = r.json()
    assert body["ncbi_meta"]["ok"] is False
    assert "503" in body["ncbi_meta"]["error"]
