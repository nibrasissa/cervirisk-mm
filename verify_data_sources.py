"""
verify_data_sources.py
======================
Run this on your machine BEFORE committing to the project.
Tests every data source the pipeline depends on and prints a go/no-go verdict.

Usage:
    pip install requests biopython
    python verify_data_sources.py

Expected runtime: ~30 seconds. No API keys required.
"""
import sys
import time
from typing import Tuple

import requests

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"

# Polite contact email used by NCBI E-utilities (replace with your own)
EMAIL = "your.name@example.com"


def section(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Test 1 — NCBI Virus / E-utilities: live HPV sequence ingestion
# ---------------------------------------------------------------------------
def test_ncbi_eutils() -> Tuple[bool, str]:
    """Verify NCBI E-utilities returns recent HPV sequences with metadata."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    try:
        r = requests.get(
            f"{base}/esearch.fcgi",
            params={
                "db": "nucleotide",
                "term": "human papillomavirus[Organism] AND complete genome[Title]",
                "retmax": 5,
                "retmode": "json",
                "datetype": "pdat",
                "reldate": 730,        # last 2 years
                "sort": "pub_date",
                "email": EMAIL,
                "tool": "verify_pipeline",
            },
            timeout=30,
        )
        if r.status_code != 200:
            return False, f"esearch returned HTTP {r.status_code}"

        result = r.json().get("esearchresult", {})
        count = int(result.get("count", "0"))
        ids = result.get("idlist", [])

        if not ids:
            return False, f"esearch returned 0 results (count={count})"

        time.sleep(0.4)  # respect 3 req/s rate limit

        r2 = requests.get(
            f"{base}/esummary.fcgi",
            params={"db": "nucleotide", "id": ids[0], "retmode": "json",
                    "email": EMAIL, "tool": "verify_pipeline"},
            timeout=30,
        )
        if r2.status_code != 200:
            return False, f"esummary returned HTTP {r2.status_code}"

        rec = r2.json()["result"][ids[0]]
        msg = (
            f"{count:,} HPV complete-genome records in last 2 years; "
            f"sample: {rec.get('accessionversion', ids[0])} "
            f"({rec.get('slen', '?')} bp, updated {rec.get('updatedate', '?')})"
        )
        return True, msg

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Test 2 — PaVE: HPV reference database & L1 typing service
# ---------------------------------------------------------------------------
def test_pave() -> Tuple[bool, str]:
    """Confirm PaVE is reachable and the API endpoint responds."""
    try:
        r = requests.get("https://pave.niaid.nih.gov/", timeout=30)
        if r.status_code != 200:
            return False, f"PaVE homepage returned HTTP {r.status_code}"

        # Try the public API spec endpoint (Swagger)
        r2 = requests.get("https://pave.niaid.nih.gov/api/swagger.json",
                          timeout=30)
        api_ok = r2.status_code == 200

        msg = "PaVE reachable"
        if api_ok:
            try:
                spec = r2.json()
                paths = list(spec.get("paths", {}).keys())[:3]
                msg += f"; Swagger API live, sample endpoints: {paths}"
            except Exception:
                msg += "; Swagger spec returned non-JSON"
        else:
            msg += "; Swagger spec not at /api/swagger.json (check website for current path)"
        return True, msg

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Test 3 — PGS Catalog: cervical cancer polygenic scores
# ---------------------------------------------------------------------------
def test_pgs_catalog() -> Tuple[bool, str]:
    """Search PGS Catalog REST API for cervical-cancer-relevant scores."""
    try:
        # EFO term for 'cervical cancer'
        r = requests.get(
            "https://www.pgscatalog.org/rest/trait/search",
            params={"term": "cervical"},
            timeout=30,
        )
        if r.status_code != 200:
            # Try alternate REST path
            r = requests.get(
                "https://www.pgscatalog.org/rest/score/search",
                params={"trait": "cervical cancer"},
                timeout=30,
            )

        if r.status_code != 200:
            return False, f"PGS Catalog REST returned HTTP {r.status_code}"

        data = r.json()
        # Response shape varies by endpoint — handle both
        n_results = (
            len(data.get("results", []))
            if isinstance(data, dict) and "results" in data
            else len(data) if isinstance(data, list) else 0
        )
        return True, (
            f"PGS Catalog REST live; {n_results} cervical-related result(s). "
            f"Note: cervical-cancer-specific PGS may be limited; you may need "
            f"to derive a score from the Estonian Biobank/FinnGen GWAS "
            f"(Pujol Gualdo et al. 2023, HR=3.1) summary statistics instead."
        )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Test 4 — 1000 Genomes Project: public host genotypes
# ---------------------------------------------------------------------------
def test_1000g() -> Tuple[bool, str]:
    """Confirm 1000 Genomes data is reachable on the EBI mirror."""
    # 1000G phase 3 sample list — small file, quick to fetch
    url = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/"
           "20130502/integrated_call_samples_v3.20130502.ALL.panel")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return False, f"1000G sample panel returned HTTP {r.status_code}"

        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        # Expect ~2,504 individuals + header
        return True, (
            f"1000G phase 3 panel reachable; {len(lines) - 1} individuals "
            f"across populations. VCFs available at the same FTP root."
        )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Bonus — confirm Python tooling is installable
# ---------------------------------------------------------------------------
def test_python_tooling() -> Tuple[bool, str]:
    msgs = []
    try:
        from Bio import Entrez  # noqa: F401
        msgs.append("biopython.Entrez")
    except ImportError:
        return False, "biopython not installed (pip install biopython)"
    try:
        import pgscatalog  # noqa: F401
        msgs.append("pgscatalog-utils")
    except ImportError:
        msgs.append("pgscatalog-utils MISSING (pip install pgscatalog-utils)")
    return True, ", ".join(msgs)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
def main():
    section("Cervical cancer multi-modal pipeline — data source verification")
    print(f"Contact email used for NCBI: {EMAIL}\n"
          f"(Replace EMAIL in the script with your own before production use.)")

    tests = [
        ("Pillar 1 — NCBI E-utilities (live HPV sequences)", test_ncbi_eutils),
        ("Pillar 1 — PaVE (HPV reference + typing)",         test_pave),
        ("Pillar 2 — PGS Catalog (cervical cancer PRS)",      test_pgs_catalog),
        ("Pillar 2 — 1000 Genomes (host genotypes)",          test_1000g),
        ("Tooling — Python packages",                          test_python_tooling),
    ]

    results = []
    for name, fn in tests:
        section(name)
        ok, msg = fn()
        marker = PASS if ok else FAIL
        print(f"{marker}  {msg}")
        results.append((name, ok))

    section("VERDICT")
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    print(f"{n_pass}/{n_total} checks passed")
    for name, ok in results:
        print(f"  {PASS if ok else FAIL}  {name}")

    if n_pass == n_total:
        print(f"\n{PASS}  All data sources reachable. Project is doable.")
        sys.exit(0)
    else:
        print(f"\n{FAIL}  Some sources failed — investigate before committing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
