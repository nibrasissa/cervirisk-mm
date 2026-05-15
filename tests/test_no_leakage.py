"""Data integrity tests for CerviRisk-MM.

These tests are the contract that training and test data are never mixed,
and that augmented features are stable across runs. They run every time
you execute `pytest tests/`.

If any future change breaks these guarantees, the test suite fails loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneOut, train_test_split


# ---------------------------------------------------------------------------
# Leakage assertions
# ---------------------------------------------------------------------------
def test_train_test_indices_are_disjoint_loocv():
    """Every LOOCV fold must have a single held-out index not in training."""
    from src.model.train import assert_no_leakage

    n = 50
    X = pd.DataFrame({"a": np.arange(n)})
    y = np.random.RandomState(0).randint(0, 2, n)

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        assert_no_leakage(train_idx, test_idx, X)
        assert len(test_idx) == 1
        assert test_idx[0] not in train_idx


def test_train_test_indices_are_disjoint_split():
    """Stratified split must produce disjoint train/test index sets."""
    from src.model.train import assert_no_leakage

    n = 200
    X = pd.DataFrame({"a": np.arange(n)})
    y = np.array([0] * 180 + [1] * 20)

    X_tr, X_te, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    assert_no_leakage(X_tr.index.to_numpy(), X_te.index.to_numpy(), X)


def test_leakage_assertion_catches_overlap():
    """Sanity: if we deliberately overlap indices, the assertion must fire."""
    from src.model.train import assert_no_leakage

    X = pd.DataFrame({"a": np.arange(10)})
    train_idx = np.array([0, 1, 2, 3, 4])
    test_idx = np.array([4, 5])  # 4 appears in both

    with pytest.raises(AssertionError, match="DATA LEAKAGE"):
        assert_no_leakage(train_idx, test_idx, X)


# ---------------------------------------------------------------------------
# Augmentation stability — same patient must get the same augmented features
# every time the assembly runs.
# ---------------------------------------------------------------------------
def test_strain_assignment_is_stable_per_patient_id():
    """Calling assemble_patients twice must yield identical strain assignments.

    Skipped if the raw UCI parquet hasn't been ingested yet (CI-friendly).
    """
    pytest.importorskip("ucimlrepo")
    from src.storage.paths import uci_raw_path

    if not uci_raw_path().exists():
        pytest.skip("UCI data not ingested; run `python -m src.ingestion.uci` first")

    from src.features.build_patient import assemble_patients

    df1 = assemble_patients()
    df2 = assemble_patients()
    pd.testing.assert_series_equal(
        df1["assigned_hpv_strain"], df2["assigned_hpv_strain"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        df1["host_prs"], df2["host_prs"], check_names=False,
    )
    pd.testing.assert_series_equal(
        df1["matched_genome_id"], df2["matched_genome_id"], check_names=False,
    )


# ---------------------------------------------------------------------------
# Outcome immutability — biopsy is never overwritten after assembly.
# ---------------------------------------------------------------------------
def test_outcome_is_not_modified_by_augmentation():
    """The biopsy outcome from UCI must equal the assembled biopsy_outcome."""
    pytest.importorskip("ucimlrepo")
    from src.storage.paths import uci_raw_path

    if not uci_raw_path().exists():
        pytest.skip("UCI data not ingested")

    from src.features.build_patient import assemble_patients

    uci_raw = pd.read_parquet(uci_raw_path())
    assembled = assemble_patients()

    # Same length, same order, same values.
    assert len(assembled) == len(uci_raw)
    uci_biopsy = pd.to_numeric(uci_raw["Biopsy"], errors="coerce").fillna(0).astype(int).values
    asm_biopsy = assembled["biopsy_outcome"].fillna(0).astype(int).values
    np.testing.assert_array_equal(uci_biopsy, asm_biopsy)
