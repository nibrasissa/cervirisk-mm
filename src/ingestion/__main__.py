"""Run all four ingestion modules in sequence.

Usage:
    python -m src.ingestion

UCI and 1000 Genomes panel are mandatory. NCBI is mandatory (the live feed).
PGS ingestion is best-effort on day 1; if no cervical-specific score exists
we proceed and address it on day 2.
"""
from __future__ import annotations

import logging
import sys

from src.ingestion import genomes_1kg, ncbi, pgs, uci

logger = logging.getLogger(__name__)


def run_all() -> int:
    """Returns process exit code: 0 if mandatory sources succeeded, 1 otherwise."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    failures: list[str] = []

    # 1. UCI — mandatory anchor
    try:
        logger.info("=" * 70)
        logger.info("STEP 1/4 — UCI dataset (anchor)")
        logger.info("=" * 70)
        uci.main()
    except Exception as e:
        logger.exception("UCI ingestion failed: %s", e)
        failures.append("uci")

    # 2. 1000 Genomes panel — mandatory for ancestry-aware augmentation
    try:
        logger.info("=" * 70)
        logger.info("STEP 2/4 — 1000 Genomes panel (host genomes index)")
        logger.info("=" * 70)
        genomes_1kg.main()
    except Exception as e:
        logger.exception("1000G ingestion failed: %s", e)
        failures.append("1000g")

    # 3. NCBI — mandatory for the live HPV signal
    try:
        logger.info("=" * 70)
        logger.info("STEP 3/4 — NCBI HPV sequences (live feed)")
        logger.info("=" * 70)
        ncbi.main()
    except Exception as e:
        logger.exception("NCBI ingestion failed: %s", e)
        failures.append("ncbi")

    # 4. PGS Catalog — best-effort on day 1
    try:
        logger.info("=" * 70)
        logger.info("STEP 4/4 — PGS Catalog scoring file (best-effort)")
        logger.info("=" * 70)
        pgs.main()
    except Exception as e:
        logger.exception("PGS ingestion failed (non-fatal on day 1): %s", e)
        # not appended to failures — PGS is best-effort on day 1

    logger.info("=" * 70)
    if failures:
        logger.error("Mandatory ingestion failures: %s", ", ".join(failures))
        return 1
    logger.info("Ingestion complete. Check data/raw/ for output files.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
