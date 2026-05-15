"""Live NCBI fetch — recent HPV sequence deposits from E-utilities.

This module pulls HPV-typed nucleotide records deposited in the last N
days, returning a small DataFrame suitable for live drift checking.

Unlike `src/ingestion/ncbi.py` (which writes a frozen parquet snapshot on
day-1 ingestion), this module is intended to be called from the live API
endpoint and returns a fresh DataFrame in memory. Caching happens at the
API layer (5-minute TTL) so NCBI is not hit on every request.

Failure mode
------------
NCBI E-utilities can rate-limit or return empty/malformed XML during
outages. This module:
  - retries once with a 2-second backoff on transient errors
  - returns an empty DataFrame (NOT raising) on persistent failure
  - logs the failure reason so the API layer can include it in the response

Run with (for manual inspection):
    python -m src.ingestion.ncbi_live
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from Bio import Entrez

logger = logging.getLogger(__name__)

# NCBI requires an email so they can contact you if your script misbehaves.
# We use a generic project address; replace with your own when deploying.
Entrez.email = "cervirisk-mm@example.org"

DEFAULT_QUERY_DAYS = 30
DEFAULT_MAX_RECORDS = 200


def _build_query(days: int) -> str:
    """Build the E-utilities search query for recent HPV deposits.

    The PDAT field is publication date — the date NCBI posted the record.
    For HPV surveillance we want recently deposited sequences regardless
    of when they were collected.
    """
    return (
        '("Human papillomavirus"[Organism] OR "HPV"[All Fields]) '
        f'AND "last {days} days"[PDAT]'
    )


def _do_search(query: str, max_records: int) -> list[str]:
    """E-utilities esearch — returns a list of NCBI GenBank IDs."""
    handle = Entrez.esearch(
        db="nucleotide",
        term=query,
        retmax=max_records,
        sort="pub date",
    )
    result = Entrez.read(handle)
    handle.close()
    return list(result.get("IdList", []))


def _do_summary(uid_list: list[str]) -> list[dict]:
    """E-utilities esummary — returns title + accession for each ID."""
    if not uid_list:
        return []
    handle = Entrez.esummary(db="nucleotide", id=",".join(uid_list))
    summaries = Entrez.read(handle)
    handle.close()
    return [dict(s) for s in summaries]


def fetch_recent_deposits(
    days: int = DEFAULT_QUERY_DAYS,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> tuple[pd.DataFrame, dict]:
    """Pull recent HPV deposits from NCBI.

    Returns (DataFrame, meta) where:
      - DataFrame has columns: accession, title, organism
        (empty if NCBI returned nothing or errored)
      - meta is a dict with: ok, n_records, query, error (None on success)
    """
    query = _build_query(days)
    started = time.time()
    meta: dict = {
        "ok": True,
        "query": query,
        "n_records": 0,
        "elapsed_seconds": 0.0,
        "error": None,
    }

    try:
        ids = _do_search(query, max_records)
    except Exception as e:
        # First attempt failed — retry once after a short pause
        logger.warning("NCBI esearch failed (%s), retrying once", e)
        time.sleep(2)
        try:
            ids = _do_search(query, max_records)
        except Exception as e2:
            meta.update(ok=False, error=f"esearch failed: {type(e2).__name__}: {e2}")
            meta["elapsed_seconds"] = round(time.time() - started, 2)
            return pd.DataFrame(columns=["accession", "title", "organism"]), meta

    if not ids:
        meta["elapsed_seconds"] = round(time.time() - started, 2)
        return pd.DataFrame(columns=["accession", "title", "organism"]), meta

    try:
        summaries = _do_summary(ids)
    except Exception as e:
        meta.update(ok=False, error=f"esummary failed: {type(e).__name__}: {e}")
        meta["elapsed_seconds"] = round(time.time() - started, 2)
        return pd.DataFrame(columns=["accession", "title", "organism"]), meta

    rows = []
    for s in summaries:
        rows.append({
            "accession": str(s.get("AccessionVersion", s.get("Caption", ""))),
            "title": str(s.get("Title", "")),
            "organism": "Human papillomavirus",
        })
    df = pd.DataFrame(rows)

    meta["n_records"] = len(df)
    meta["elapsed_seconds"] = round(time.time() - started, 2)
    logger.info("Fetched %d HPV deposits from NCBI in %.2fs",
                len(df), meta["elapsed_seconds"])
    return df, meta


def main() -> None:
    """CLI: print the most recent HPV deposits to stdout."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    df, meta = fetch_recent_deposits()
    print(f"\nLive NCBI fetch — query: {meta['query']}")
    print(f"Records: {meta['n_records']}   elapsed: {meta['elapsed_seconds']}s")
    if not meta["ok"]:
        print(f"ERROR: {meta['error']}")
        return
    if df.empty:
        print("(no deposits matched)")
        return
    print("\nFirst 10 records:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
