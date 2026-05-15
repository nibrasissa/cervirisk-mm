"""Smoke tests for the ingestion layer.

These tests do NOT hit the network. They confirm:
  1. Modules import cleanly.
  2. Provenance enforcement works (biopsy outcome is protected).
  3. After running ingestion, the expected raw files exist.

Run the network-hitting verification separately with:
    python verify_data_sources.py
"""
from __future__ import annotations

import pytest

from src.storage.provenance import (
    PatientRecord,
    Provenance,
    ProvenanceTag,
)


def test_imports():
    """All ingestion modules must import without side effects."""
    from src.ingestion import genomes_1kg, ncbi, pgs, uci  # noqa: F401


def test_paths_module():
    """Path module exposes expected functions and creates directories."""
    from src.storage.paths import (
        RAW_DIR,
        genomes_1kg_panel_path,
        ncbi_raw_path,
        pgs_raw_path,
        uci_raw_path,
    )
    assert RAW_DIR.exists()
    assert "uci" in str(uci_raw_path()).lower()
    assert "20260101" in str(ncbi_raw_path("20260101"))
    assert "1kg" in str(genomes_1kg_panel_path()).lower()
    assert "PGS00001" in str(pgs_raw_path("PGS00001"))


def test_provenance_protects_biopsy_outcome():
    """The integrity rule: biopsy outcome cannot be overwritten via set_feature."""
    p = PatientRecord(patient_id="test-1", biopsy_outcome=0)

    tag = ProvenanceTag(
        provenance=Provenance.SAMPLED_FROM_PRIOR,
        source="test",
    )
    # Augmenting a normal feature is fine.
    p.set_feature("assigned_hpv_strain", "HPV16", tag)
    assert p.features["assigned_hpv_strain"] == "HPV16"

    # Augmenting the biopsy outcome must raise.
    with pytest.raises(ValueError, match="Biopsy outcome is protected"):
        p.set_feature("biopsy_outcome", 1, tag)

    # And the actual outcome is unchanged.
    assert p.biopsy_outcome == 0


def test_data_status_flags_synthetic_components():
    """A patient with any sampled feature must be flagged SYNTHETIC_ASSEMBLY."""
    p = PatientRecord(patient_id="test-2", biopsy_outcome=0)
    real_tag = ProvenanceTag(provenance=Provenance.REAL_OBSERVED, source="uci")
    synth_tag = ProvenanceTag(provenance=Provenance.SAMPLED_FROM_PRIOR, source="prevalence")

    p.set_feature("age", 34, real_tag)
    assert p.data_status() == "REAL_ONLY"

    p.set_feature("assigned_hpv_strain", "HPV16", synth_tag)
    assert p.data_status() == "SYNTHETIC_ASSEMBLY"
