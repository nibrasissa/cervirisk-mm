"""Ingest the 1000 Genomes Project sample panel.

The panel is a small TSV (~150 KB) listing all ~2,500 phase-3 individuals
with their population and super-population labels. We use it on day 2 to
draw ancestry-matched genomes for synthetic patient assembly.

The actual VCF genotype files are much larger and downloaded on-demand later
(only the SNPs that appear in the PGS scoring file are pulled, via tabix).
For day 1, the panel alone is what we need.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from src.storage.paths import GENOMES_1KG_PANEL_URL, genomes_1kg_panel_path

logger = logging.getLogger(__name__)


# Super-population mapping used when stratifying genomes by ancestry.
# The Caracas-cohort UCI patients are presumed AMR (Admixed American).
SUPER_POP_LABELS = {
    "AFR": "African",
    "AMR": "Admixed American",
    "EAS": "East Asian",
    "EUR": "European",
    "SAS": "South Asian",
}


def fetch_panel() -> pd.DataFrame:
    logger.info("Fetching 1000 Genomes phase-3 panel from %s", GENOMES_1KG_PANEL_URL)
    r = requests.get(GENOMES_1KG_PANEL_URL, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    # Normalize column names — the panel sometimes uses 'gender' and trailing whitespace.
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns: sample, pop, super_pop, gender
    expected = {"sample", "pop", "super_pop", "gender"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"1000G panel missing expected columns: {missing}")

    # Drop empty rows (some panels have a trailing blank line).
    df = df[df["sample"].notna() & (df["sample"] != "")].reset_index(drop=True)

    logger.info("1000G panel loaded: %d individuals across %d populations",
                len(df), df["pop"].nunique())
    return df


def save_panel(df: pd.DataFrame) -> None:
    out = genomes_1kg_panel_path()
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))


def summarize(df: pd.DataFrame) -> None:
    """Print population breakdown for sanity-checking."""
    summary = df.groupby("super_pop").size().sort_values(ascending=False)
    logger.info("Super-population counts:")
    for pop, n in summary.items():
        label = SUPER_POP_LABELS.get(pop, pop)
        logger.info("  %s (%s): %d", pop, label, n)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    df = fetch_panel()

    # Sanity checks — the contract of the 1000G panel.
    assert len(df) > 2000, f"expected ~2,500 individuals, got {len(df)}"
    assert "AMR" in df["super_pop"].unique(), "AMR super-population missing"

    summarize(df)
    save_panel(df)


if __name__ == "__main__":
    main()
