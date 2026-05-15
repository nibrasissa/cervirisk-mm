"""Ingest recent HPV sequence metadata from NCBI E-utilities.

This is the "live" data source for CerviRisk-MM. Labs worldwide deposit HPV
sequences (complete genomes, L1 gene fragments for typing, variants) and we
pull recent submissions so the pipeline reflects current circulating-strain
composition.

Reference: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Rate limits: 3 requests/second without API key, 10/sec with one. We respect
both by sleeping between calls. Set NCBI_API_KEY environment variable to use
your key.

Query strategy
--------------
The original query required "complete genome" in the title — too restrictive.
Most HPV submissions are partial sequences (the L1 gene used for typing, or
specific variants). We search the broader Organism field and let the feature
engineering layer filter to what it needs.

If a recent window yields zero results, we expand the window automatically
through a fallback ladder so the pipeline self-heals during quiet periods.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
import requests

from src.storage.paths import (
    NCBI_API_KEY,
    NCBI_CONTACT_EMAIL,
    NCBI_EUTILS_BASE,
    TOOL_NAME,
    ncbi_raw_path,
)

logger = logging.getLogger(__name__)


# Rate limits — sleep between requests to be polite to NCBI's servers.
_REQ_DELAY_S = 0.12 if NCBI_API_KEY else 0.35

# Search query — broad enough to capture typing-relevant sequences. We accept
# all HPV-organism records and filter downstream.
HPV_SEARCH_TERM = "human papillomavirus[Organism]"

# Window-expansion ladder. If 365 days yields nothing, try 730, then 1095.
# This keeps the pipeline robust during quiet submission periods.
_FALLBACK_WINDOWS_DAYS = (365, 730, 1095, 1825)  # 1y → 2y → 3y → 5y


def _polite_params() -> dict[str, str]:
    p = {"email": NCBI_CONTACT_EMAIL, "tool": TOOL_NAME}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


def _esearch(reldate_days: int, retmax: int) -> tuple[list[str], int]:
    """Run a single esearch and return (uids, total_matching_count)."""
    url = f"{NCBI_EUTILS_BASE}/esearch.fcgi"
    params = {
        **_polite_params(),
        "db": "nucleotide",
        "term": HPV_SEARCH_TERM,
        "retmax": str(retmax),
        "retmode": "json",
        "datetype": "pdat",
        "reldate": str(reldate_days),
        "sort": "pub_date",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    result = r.json().get("esearchresult", {})
    ids = result.get("idlist", [])
    total = int(result.get("count", "0"))
    time.sleep(_REQ_DELAY_S)
    return ids, total


def search_recent_hpv(reldate_days: int = 365, retmax: int = 200) -> list[str]:
    """Find HPV nucleotide records submitted recently, with auto-expanding window.

    Starts at `reldate_days`; if zero results, expands through the fallback
    ladder until something is found or the ladder is exhausted.
    """
    windows = [reldate_days] + [w for w in _FALLBACK_WINDOWS_DAYS if w > reldate_days]
    last_total = 0

    for window in windows:
        logger.info("NCBI esearch: %s, last %d days, retmax=%d",
                    HPV_SEARCH_TERM, window, retmax)
        ids, total = _esearch(reldate_days=window, retmax=retmax)
        last_total = total
        if ids:
            logger.info("NCBI esearch returned %d ids (of %d total matching, "
                        "window=%d days)", len(ids), total, window)
            return ids
        logger.warning("Empty result at window=%d days; expanding…", window)

    logger.error("Exhausted fallback windows — NCBI returned 0 HPV records "
                 "even at the widest window. Total matching: %d", last_total)
    return []


def fetch_summaries(uids: list[str], batch_size: int = 100) -> list[dict[str, Any]]:
    """Fetch summary metadata for a list of NCBI nucleotide UIDs."""
    if not uids:
        return []

    url = f"{NCBI_EUTILS_BASE}/esummary.fcgi"
    records: list[dict[str, Any]] = []

    for i in range(0, len(uids), batch_size):
        chunk = uids[i : i + batch_size]
        params = {**_polite_params(), "db": "nucleotide",
                  "id": ",".join(chunk), "retmode": "json"}
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        result = r.json().get("result", {})
        uids_returned = result.get("uids", [])

        for uid in uids_returned:
            doc = result.get(uid, {})
            records.append({
                "uid": uid,
                "accession": doc.get("accessionversion") or doc.get("caption", ""),
                "title": doc.get("title", ""),
                "organism": doc.get("organism", ""),
                "length_bp": int(doc.get("slen", 0) or 0),
                "submission_date": doc.get("createdate", ""),
                "update_date": doc.get("updatedate", ""),
                "taxid": doc.get("taxid", ""),
            })

        logger.info("NCBI esummary: fetched %d/%d", len(records), len(uids))
        time.sleep(_REQ_DELAY_S)

    return records


def enrich_with_country(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull GenBank flat records to extract country-of-origin metadata.

    esummary doesn't include geographic info, so we use efetch to retrieve
    GenBank-format records and parse the /country=... qualifier. Best-effort:
    not all GenBank records have country metadata, and the call is the slowest
    in this module, so we cap the enrichment at 500 records by default.
    """
    if not records:
        return records

    # Cap enrichment for performance — full enrichment can be slow with many records.
    to_enrich = records[:500]
    uids = [r["uid"] for r in to_enrich]
    url = f"{NCBI_EUTILS_BASE}/efetch.fcgi"
    params = {**_polite_params(), "db": "nucleotide",
              "id": ",".join(uids), "rettype": "gb", "retmode": "xml"}
    logger.info("NCBI efetch (gb): enriching %d records with country data", len(uids))
    try:
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Country enrichment failed (non-fatal): %s", e)
        for rec in records:
            rec["country"] = ""
            rec["isolation_source"] = ""
        return records
    time.sleep(_REQ_DELAY_S)

    # Parse GBSeq XML — index by accession for fast lookup.
    countries: dict[str, str] = {}
    isolation_sources: dict[str, str] = {}
    try:
        root = ET.fromstring(r.content)
        for seq in root.findall(".//GBSeq"):
            acc_elem = seq.find("GBSeq_accession-version")
            if acc_elem is None or acc_elem.text is None:
                continue
            acc = acc_elem.text
            for qual in seq.findall(".//GBQualifier"):
                name = qual.findtext("GBQualifier_name", "")
                value = qual.findtext("GBQualifier_value", "")
                if name == "country":
                    countries[acc] = value
                elif name == "isolation_source":
                    isolation_sources[acc] = value
    except ET.ParseError as e:
        logger.warning("Failed to parse efetch XML (non-fatal): %s", e)

    for rec in records:
        rec["country"] = countries.get(rec["accession"], "")
        rec["isolation_source"] = isolation_sources.get(rec["accession"], "")

    found = sum(1 for r in records if r["country"])
    logger.info("Country metadata available for %d/%d records", found, len(records))
    return records


def ingest(reldate_days: int = 365, retmax: int = 200) -> pd.DataFrame:
    """End-to-end ingestion: search, summarize, enrich, return DataFrame."""
    uids = search_recent_hpv(reldate_days=reldate_days, retmax=retmax)
    if not uids:
        logger.warning("No HPV records returned from NCBI at any window")
        return pd.DataFrame()

    records = fetch_summaries(uids)
    records = enrich_with_country(records)
    df = pd.DataFrame(records)

    df["ingested_at"] = dt.datetime.utcnow().isoformat()
    return df


def save_ncbi(df: pd.DataFrame, date_str: str | None = None) -> None:
    if date_str is None:
        date_str = dt.datetime.utcnow().strftime("%Y%m%d")
    out = ncbi_raw_path(date_str)
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    df = ingest(reldate_days=365, retmax=200)
    if df.empty:
        raise SystemExit("NCBI returned 0 HPV records even after window expansion")
    save_ncbi(df)


if __name__ == "__main__":
    main()
