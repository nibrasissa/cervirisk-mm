"""Ingest a polygenic scoring file from PGS Catalog.

PGS Catalog provides published polygenic scores as TSV files containing one
row per variant with effect alleles and effect weights. We download the
scoring file once and use it later to compute PRS for 1000 Genomes
individuals.

The catalog's REST API is documented at:
    https://www.pgscatalog.org/rest/
Files are hosted on the EBI FTP server.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from src.storage.paths import PGS_CATALOG_REST, pgs_raw_path

logger = logging.getLogger(__name__)


# Trait EFO IDs to search for. The catalog uses the EBI Experimental Factor
# Ontology. EFO_0001416 = "cervical cancer"; we'll also accept related uterine
# cancer scores as fallback for the day-1 ingestion.
TARGET_EFO_IDS = ("EFO_0001416",)
FALLBACK_KEYWORD = "cervical"


def find_scores_for_trait(efo_id: str) -> list[dict]:
    """Query PGS Catalog REST for scoring files associated with an EFO trait."""
    url = f"{PGS_CATALOG_REST}/trait/{efo_id}"
    logger.info("PGS Catalog: querying trait %s", efo_id)
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        logger.warning("EFO %s not found in PGS Catalog", efo_id)
        return []
    r.raise_for_status()
    data = r.json()
    return data.get("associated_pgs_ids", []) or []


def search_scores_by_keyword(keyword: str) -> list[str]:
    """Fallback: search for scores by keyword across all traits."""
    url = f"{PGS_CATALOG_REST}/score/search"
    params = {"trait": keyword}
    logger.info("PGS Catalog: keyword search for '%s'", keyword)
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    return [item.get("id", "") for item in results if item.get("id")]


def fetch_score_metadata(pgs_id: str) -> dict:
    """Fetch full metadata for one scoring file by its PGS ID."""
    url = f"{PGS_CATALOG_REST}/score/{pgs_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def download_scoring_file(pgs_id: str, ftp_url: str) -> pd.DataFrame:
    """Download the scoring TSV from the EBI FTP server.

    PGS scoring files are gzipped TSVs with a header block of '#' lines
    followed by tab-separated variant data. Columns vary but typically
    include: chr_name, chr_position, effect_allele, other_allele, effect_weight.
    """
    logger.info("Downloading PGS scoring file: %s", ftp_url)
    r = requests.get(ftp_url, timeout=120)
    r.raise_for_status()

    # Save the raw .txt.gz to disk first, then read it.
    out = pgs_raw_path(pgs_id)
    out.write_bytes(r.content)

    # Parse it — pandas reads gzipped TSV directly, '#' marks comment lines.
    df = pd.read_csv(out, sep="\t", comment="#", compression="infer")
    logger.info("PGS %s: %d variants loaded", pgs_id, len(df))
    return df


def ingest() -> tuple[str, pd.DataFrame] | None:
    """End-to-end: find a cervical-cancer-relevant PGS, download it, parse it.

    Returns (pgs_id, dataframe) on success, or None if nothing relevant found.
    """
    # 1. Try the canonical EFO IDs
    candidate_ids: list[str] = []
    for efo in TARGET_EFO_IDS:
        candidate_ids.extend(find_scores_for_trait(efo))

    # 2. Fallback to keyword search
    if not candidate_ids:
        logger.info("No scores on canonical EFO; trying keyword fallback")
        candidate_ids = search_scores_by_keyword(FALLBACK_KEYWORD)

    if not candidate_ids:
        logger.warning(
            "No cervical-cancer-specific PGS found. The pipeline will need a "
            "manually-constructed score from GWAS summary statistics (e.g. "
            "Pujol Gualdo et al. 2023). Skipping PGS ingestion for now."
        )
        return None

    # 3. Pick the first one — for day 1 we just need *a* score to flow through.
    pgs_id = candidate_ids[0]
    logger.info("Selected PGS %s (of %d candidates)", pgs_id, len(candidate_ids))

    meta = fetch_score_metadata(pgs_id)
    ftp_link = meta.get("ftp_scoring_file", "")
    if not ftp_link:
        logger.error("PGS %s has no ftp_scoring_file link in metadata", pgs_id)
        return None

    df = download_scoring_file(pgs_id, ftp_link)
    return pgs_id, df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    result = ingest()
    if result is None:
        # Not a fatal failure — day 1 can proceed without PGS; we'll address
        # this on day 2 either by using a related-cancer PGS as placeholder
        # or constructing a score from GWAS summary statistics.
        logger.warning("PGS ingestion produced no file (see message above)")
        return
    pgs_id, df = result
    logger.info("PGS ingestion complete: %s, %d variants", pgs_id, len(df))


if __name__ == "__main__":
    main()
