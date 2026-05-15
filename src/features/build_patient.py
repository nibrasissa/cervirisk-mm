"""Patient assembly: take a UCI row, augment it into a feature vector.

This is the integration logic. The principles enforced here:

  1. ANCHOR — every patient starts as a real UCI row. UCI columns are
     preserved unchanged.
  2. AUGMENT — additional features (strain, carcinogenicity, host PRS) are
     added with explicit provenance tags.
  3. NEVER OVERWRITE OUTCOMES — biopsy_outcome is set once from UCI and
     protected.
  4. NO RECORD JOINS — UCI rows, 1000G individuals, and HPV strains are
     not linked in any source. The assembly creates synthetic linkages that
     are clearly labeled and provenance-tagged.
  5. STABILITY ACROSS RUNS — strain and genome assignments are deterministic
     per patient_id. This prevents the same UCI patient being assigned a
     different HPV strain on every LOOCV fold, which would otherwise inject
     noise the model cannot learn.

UCI HPV reporting nuance
------------------------
UCI has two HPV-evidence columns: STDs:HPV (history of HPV as an STD) and
Dx:HPV (clinical diagnosis flag). The STDs columns are CONDITIONAL on STDs
being filled — many rows have NaN even when the patient may have had HPV.
We therefore take the UNION of both indicators. We deliberately do NOT use
biopsy_outcome — that would leak the model's target into the features.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from src.features.host_prs import build_prs_lookup
from src.features.strain import (
    PUBLISHED_PREVALENCE,
    lookup_carcinogenicity,
)
from src.storage.paths import PROCESSED_DIR, genomes_1kg_panel_path, uci_raw_path

logger = logging.getLogger(__name__)


# UCI columns that go straight into the feature vector unchanged.
UCI_FEATURE_COLUMNS = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs:HPV",
    "STDs:HIV",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
]
UCI_OUTCOME_COLUMNS = ["Hinselmann", "Schiller", "Citology", "Biopsy"]

# Of the four outcome columns, only Biopsy is the prediction target.
# The other three are intermediate diagnostic test results (visual screening,
# iodine staining, Pap smear) that occur BEFORE biopsy in standard clinical
# workflow. They can be used as features in a "triage" model that predicts
# biopsy outcome given prior screening results — a different but clinically
# real question from "predict outcome given just risk factors".
SCREENING_TEST_COLUMNS = ["Hinselmann", "Schiller", "Citology"]

HPV_EVIDENCE_COLUMNS = ["STDs:HPV", "Dx:HPV"]
UCI_PRESUMED_SUPER_POP = "AMR"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in UCI_FEATURE_COLUMNS + UCI_OUTCOME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Age"]).reset_index(drop=True)
    return df


def _identify_hpv_evidence(df: pd.DataFrame) -> pd.Series:
    available = [c for c in HPV_EVIDENCE_COLUMNS if c in df.columns]
    if not available:
        logger.warning("No HPV indicator columns found in UCI")
        return pd.Series([False] * len(df), index=df.index)

    flags = df[available].fillna(0).astype(int)
    mask = (flags > 0).any(axis=1)

    per_col = {c: int(flags[c].sum()) for c in available}
    logger.info("HPV evidence per column: %s | union: %d patients",
                per_col, int(mask.sum()))
    return mask


def _stable_seed(patient_id: str, salt: str) -> int:
    """Deterministic 32-bit seed from a patient ID + salt.

    Used so the same patient gets the same strain and the same matched genome
    every time assemble_patients() runs — across all LOOCV folds, runs, and
    machines. Without this, each fold would re-sample, injecting noise.
    """
    h = hashlib.md5(f"{patient_id}:{salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _stable_strain_for(patient_id: str) -> str:
    """Deterministic HPV strain for one patient_id, sampled once and stable forever."""
    rng = np.random.default_rng(_stable_seed(patient_id, "strain"))
    strains = list(PUBLISHED_PREVALENCE.keys())
    probs = list(PUBLISHED_PREVALENCE.values())
    return rng.choice(strains, p=probs)


def _stable_genome_for(patient_id: str, pool: pd.DataFrame) -> pd.Series:
    """Deterministic 1000G individual for one patient_id from a pool of choices."""
    rng = np.random.default_rng(_stable_seed(patient_id, "genome"))
    idx = int(rng.integers(0, len(pool)))
    return pool.iloc[idx]


def assemble_patients(seed: int = 42) -> pd.DataFrame:
    """Build augmented patient table.

    The `seed` parameter is preserved for compatibility but no longer
    controls per-patient assignments — those are stable per patient_id.
    The seed only affects the order in which we iterate the AMR pool;
    final assignments are deterministic per patient regardless.
    """
    from src.storage.files import require_file

    # 1. ANCHOR
    require_file(
        uci_raw_path(),
        hint_command="python -m src.ingestion",
        description="UCI Cervical Cancer Risk Factors parquet from day-1 ingestion",
    )
    require_file(
        genomes_1kg_panel_path(),
        hint_command="python -m src.ingestion",
        description="1000 Genomes phase-3 sample panel from day-1 ingestion",
    )
    uci = pd.read_parquet(uci_raw_path())
    uci = _normalize_columns(uci)
    n_patients = len(uci)
    logger.info("Anchored on UCI: %d patients after normalization", n_patients)

    keep = [c for c in (UCI_FEATURE_COLUMNS + UCI_OUTCOME_COLUMNS) if c in uci.columns]
    df = uci[keep].copy()
    df.insert(0, "patient_id", [f"uci_{i:04d}" for i in range(n_patients)])
    if "Biopsy" in df.columns:
        df = df.rename(columns={"Biopsy": "biopsy_outcome"})

    # 2. AUGMENT — HPV strain (sampled per-patient_id, stable forever)
    hpv_pos_mask = _identify_hpv_evidence(df)
    strains = np.full(n_patients, "NONE", dtype=object)
    for i in np.where(hpv_pos_mask)[0]:
        strains[i] = _stable_strain_for(df.loc[i, "patient_id"])
    df["assigned_hpv_strain"] = strains

    n_hpv_pos = int(hpv_pos_mask.sum())
    if n_hpv_pos:
        sampled_dist = pd.Series(strains[hpv_pos_mask]).value_counts().to_dict()
        logger.info("Strain assignment (stable per patient_id): %d patients tagged; distribution: %s",
                    n_hpv_pos, sampled_dist)

    # 3. AUGMENT — carcinogenicity (deterministic lookup)
    df["strain_carcinogenicity"] = df["assigned_hpv_strain"].apply(lookup_carcinogenicity)

    # 4. AUGMENT — ancestry-matched host PRS (stable per patient_id)
    prs_table = build_prs_lookup()
    amr_pool = prs_table[prs_table["super_pop"] == UCI_PRESUMED_SUPER_POP].reset_index(drop=True)
    if len(amr_pool) == 0:
        raise ValueError(f"No 1000G individuals match super_pop={UCI_PRESUMED_SUPER_POP}")

    matched_ids, matched_pops, matched_prs = [], [], []
    for pid in df["patient_id"]:
        chosen = _stable_genome_for(pid, amr_pool)
        matched_ids.append(chosen["sample"])
        matched_pops.append(chosen["super_pop"])
        matched_prs.append(chosen["host_prs"])
    df["matched_genome_id"] = matched_ids
    df["matched_super_pop"] = matched_pops
    df["host_prs"] = matched_prs
    logger.info("PRS attached (stable per patient_id): range %.3f to %.3f (mean %.3f)",
                df["host_prs"].min(), df["host_prs"].max(), df["host_prs"].mean())

    # 5. Provenance tag
    df["data_status"] = "SYNTHETIC_ASSEMBLY"
    return df


def save_assembled(df: pd.DataFrame) -> None:
    out = PROCESSED_DIR / "patients_assembled.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows, %d columns)", out, *df.shape)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    df = assemble_patients()

    # Stability check — run twice, confirm identical strain assignment.
    df2 = assemble_patients()
    pd.testing.assert_series_equal(df["assigned_hpv_strain"], df2["assigned_hpv_strain"])
    logger.info("Stability check passed: strain assignment is deterministic per patient_id")

    save_assembled(df)
    print(f"\nAssembled {len(df)} patients with {df.shape[1]} columns.")


if __name__ == "__main__":
    main()
