"""Central paths and constants for CerviRisk-MM.

All filesystem paths and source URLs are defined here so individual modules
don't hardcode them. Override via environment variables in production.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("CERVIRISK_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("CERVIRISK_MODELS_DIR", PROJECT_ROOT / "models"))

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Raw file names — one canonical place
# ---------------------------------------------------------------------------
def uci_raw_path() -> Path:
    return RAW_DIR / "uci_cervical_cancer.parquet"


def ncbi_raw_path(date_str: str) -> Path:
    """date_str in YYYYMMDD format."""
    return RAW_DIR / f"ncbi_hpv_sequences_{date_str}.parquet"


def pgs_raw_path(pgs_id: str) -> Path:
    return RAW_DIR / f"pgs_{pgs_id}.tsv"


def genomes_1kg_panel_path() -> Path:
    return RAW_DIR / "1kg_samples.parquet"


# ---------------------------------------------------------------------------
# External API endpoints
# ---------------------------------------------------------------------------
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

PGS_CATALOG_REST = "https://www.pgscatalog.org/rest"
PGS_CATALOG_FTP = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores"

GENOMES_1KG_PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/"
    "20130502/integrated_call_samples_v3.20130502.ALL.panel"
)

UCI_DATASET_ID = 383  # Cervical Cancer (Risk Factors)


# ---------------------------------------------------------------------------
# Contact email for polite NCBI usage — override in your environment
# ---------------------------------------------------------------------------
NCBI_CONTACT_EMAIL = os.environ.get(
    "NCBI_CONTACT_EMAIL", "cervirisk-mm@example.com"
)
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")  # optional, raises rate limit
TOOL_NAME = "cervirisk-mm"
