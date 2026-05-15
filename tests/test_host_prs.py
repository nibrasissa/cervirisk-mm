"""Tests for the host polygenic risk score module (day-4 upgrade).

Verifies:
  - PRS computation is deterministic per sample_id
  - PRS distributions differ by super-population (driven by allele frequencies)
  - PRS values are bounded in a biologically reasonable range
  - The variant table has the documented structure
  - All super-populations are covered in every variant's frequency table
"""
from __future__ import annotations

import pytest


def test_variant_table_structure():
    """Every variant must declare rsid, locus, chr, weight, and freq for 5 pops."""
    from src.features.host_prs import CERVICAL_CANCER_PRS_VARIANTS

    assert len(CERVICAL_CANCER_PRS_VARIANTS) >= 10, "should have ≥10 variants"

    for v in CERVICAL_CANCER_PRS_VARIANTS:
        assert v.rsid.startswith("rs"), f"bad rsid: {v.rsid}"
        assert v.locus, f"missing locus for {v.rsid}"
        assert v.chromosome, f"missing chromosome for {v.rsid}"
        assert isinstance(v.log_or_weight, float)
        # Cervical cancer GWAS effect sizes are modest; reject anything implausible
        assert 0 < v.log_or_weight < 1.0, (
            f"log_OR for {v.rsid} = {v.log_or_weight} outside plausible range"
        )

        # Every super-population must have a frequency
        for pop in ("EUR", "AMR", "AFR", "EAS", "SAS"):
            assert pop in v.freq_by_super_pop, f"{v.rsid} missing {pop}"
            p = v.freq_by_super_pop[pop]
            assert 0 < p < 1, f"{v.rsid} {pop} frequency {p} out of (0,1)"


def test_prs_is_deterministic_per_sample_id():
    """The same sample_id must return exactly the same PRS, every call."""
    from src.features.host_prs import compute_prs

    p1 = compute_prs("HG00001", "EUR")
    p2 = compute_prs("HG00001", "EUR")
    p3 = compute_prs("HG00001", "EUR")
    assert p1 == p2 == p3


def test_prs_differs_by_individual():
    """Different sample_ids should generally produce different PRS values."""
    from src.features.host_prs import compute_prs

    values = [compute_prs(f"HG{i:05d}", "EUR") for i in range(50)]
    # Allow some duplicates but not all-identical
    assert len(set(values)) > 5, (
        "PRS should vary across individuals (different genotype samples)"
    )


def test_prs_distribution_differs_by_super_pop():
    """AFR and EUR should have measurably different mean PRS due to allele frequency differences."""
    import statistics
    from src.features.host_prs import compute_prs

    n = 200
    eur = [compute_prs(f"E_{i:05d}", "EUR") for i in range(n)]
    afr = [compute_prs(f"A_{i:05d}", "AFR") for i in range(n)]

    mean_eur = statistics.mean(eur)
    mean_afr = statistics.mean(afr)
    # Difference should be detectable (allele frequencies for HLA, CHRNA, etc. differ)
    assert abs(mean_eur - mean_afr) > 0.01, (
        f"Mean PRS too similar between EUR ({mean_eur:.3f}) and AFR ({mean_afr:.3f})"
    )


def test_prs_in_biologically_reasonable_range():
    """A 10-variant PRS with log-OR weights ~0.1 should land in roughly [0, 4]."""
    from src.features.host_prs import compute_prs

    for sid in [f"S_{i:05d}" for i in range(200)]:
        for pop in ("EUR", "AMR", "AFR", "EAS", "SAS"):
            p = compute_prs(sid, pop)
            assert 0 <= p <= 5, f"PRS {p} for {sid}/{pop} outside plausible range"


def test_unknown_super_pop_falls_back_to_eur():
    """If a super_pop is unknown, the function should not crash."""
    from src.features.host_prs import compute_prs

    p_eur = compute_prs("HG99999", "EUR")
    p_unknown = compute_prs("HG99999", "UNKNOWN_POP")
    # Same sample, fallback to EUR → identical
    assert p_eur == p_unknown


def test_build_prs_lookup_smoke():
    """End-to-end: should return a DataFrame with the expected columns."""
    pytest.importorskip("pandas")
    from src.storage.paths import genomes_1kg_panel_path

    if not genomes_1kg_panel_path().exists():
        pytest.skip("1000G panel not ingested; run `python -m src.ingestion` first")

    from src.features.host_prs import build_prs_lookup

    df = build_prs_lookup()
    for col in ("sample", "pop", "super_pop", "host_prs"):
        assert col in df.columns, f"missing column: {col}"
    assert len(df) > 0
    assert df["host_prs"].notna().all()
