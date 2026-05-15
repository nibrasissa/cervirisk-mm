"""Ingest the UCI Cervical Cancer Risk Factors dataset.

Source: Fernandes, Cardoso, Fernandes (2017), Hospital Universitario de Caracas.
UCI Machine Learning Repository, ID 383. Licensed CC BY 4.0.

This is the only source providing real patient features with real biopsy
outcomes. It is the anchor of the entire pipeline. Every augmented patient
in CerviRisk-MM starts as a UCI row.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.storage.paths import UCI_DATASET_ID, uci_raw_path

logger = logging.getLogger(__name__)


# Outcome columns in the UCI dataset (one or more of these is the target).
# Biopsy is the gold-standard outcome for our risk model.
OUTCOME_COLUMNS = ("Hinselmann", "Schiller", "Citology", "Biopsy")


def fetch_uci() -> pd.DataFrame:
    """Download UCI cervical cancer risk factors dataset.

    Returns a DataFrame with all 36 columns and 858 patients. Missing values
    appear as '?' in some columns; we preserve them as NaN here and handle
    them in the feature engineering layer.
    """
    from ucimlrepo import fetch_ucirepo

    logger.info("Fetching UCI dataset id=%d", UCI_DATASET_ID)
    ds = fetch_ucirepo(id=UCI_DATASET_ID)

    # ucimlrepo returns features and targets separately; we want one wide frame.
    features = ds.data.features
    targets = ds.data.targets
    df = pd.concat([features, targets], axis=1)

    # Convert string '?' to NaN where it appears (UCI's missing convention).
    df = df.replace("?", pd.NA)

    # Coerce numeric columns; UCI ships some as strings due to the '?' markers.
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # leave genuinely non-numeric columns alone

    logger.info("UCI dataset loaded: %d rows × %d columns", *df.shape)
    return df


def save_uci(df: pd.DataFrame) -> None:
    """Write UCI dataset to data/raw/ as Parquet."""
    out = uci_raw_path()
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    df = fetch_uci()

    # Sanity checks — these are the contract of the UCI source.
    assert len(df) == 858, f"expected 858 patients, got {len(df)}"
    for outcome in OUTCOME_COLUMNS:
        assert outcome in df.columns, f"missing outcome column: {outcome}"

    n_biopsy_pos = int(df["Biopsy"].fillna(0).astype(int).sum())
    logger.info("UCI biopsy-positive count: %d / %d (%.1f%%)",
                n_biopsy_pos, len(df), 100 * n_biopsy_pos / len(df))

    save_uci(df)


if __name__ == "__main__":
    main()
