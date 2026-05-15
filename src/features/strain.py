"""HPV strain prevalence and assignment.

Two distributions live here:

1. PUBLISHED_PREVALENCE — global HPV genotype attribution in invasive cervical
   cancer and high-grade cervical disease. Sourced from de Sanjosé et al.
   (Lancet Oncology, 2010), the largest international HPV typing study, with
   IARC monograph adjustments. This is the prior used to SAMPLE strain
   assignments for HPV-positive UCI patients.

2. live_prevalence_from_ncbi() — computes the prevalence among recent
   NCBI deposits. This is NOT used for sampling (NCBI submissions are biased
   toward research interest, not population prevalence). It is used for
   drift monitoring: if the published prior diverges from the live signal,
   something epidemiologically interesting is happening (e.g. post-vaccination
   strain replacement).

Reference
---------
de Sanjosé, S., et al. (2010). Human papillomavirus genotype attribution in
    invasive cervical cancer: a retrospective cross-sectional worldwide study.
    Lancet Oncology, 11(11), 1048–1056.
"""
from __future__ import annotations

import logging
import re
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Published global prevalence in invasive cervical cancer + high-grade lesions.
# These sum to 1.0 — used as a categorical distribution for sampling.
PUBLISHED_PREVALENCE: dict[str, float] = {
    "HPV16": 0.55,
    "HPV18": 0.15,
    "HPV31": 0.05,
    "HPV33": 0.04,
    "HPV45": 0.04,
    "HPV52": 0.03,
    "HPV58": 0.03,
    "OTHER_HR_HPV": 0.10,   # 35, 39, 51, 56, 59, 68, 73, 82
    "LOW_RISK_HPV": 0.01,   # 6, 11, 40, 42, 43, 44, 53, 54, 61, 70, 72, 81
}


# IARC carcinogenicity classification, mapped to a scalar score 0–1.
# Group 1 = carcinogenic to humans (HPV16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58, 59).
# Group 2A = probably carcinogenic (68). Group 2B = possibly (26, 53, 66, 67, 70, 73, 82, 30, 34, 69, 85, 97).
# Group 3 = not classifiable. Low-risk types are non-carcinogenic.
CARCINOGENICITY: dict[str, float] = {
    "HPV16": 0.95,        # highest carcinogenic potential, ~55% of all cases
    "HPV18": 0.85,        # second highest
    "HPV31": 0.70,
    "HPV33": 0.70,
    "HPV45": 0.70,
    "HPV52": 0.65,
    "HPV58": 0.65,
    "OTHER_HR_HPV": 0.50,
    "LOW_RISK_HPV": 0.05,
}


_STRAIN_REGEX = re.compile(r"HPV[\s\-_]?(\d+)", re.IGNORECASE)


def extract_strain_from_title(title: str) -> str:
    """Parse an HPV strain from a GenBank record title.

    Returns the strain key (e.g. 'HPV16') or 'UNSPECIFIED' if no match.
    """
    if not isinstance(title, str):
        return "UNSPECIFIED"
    m = _STRAIN_REGEX.search(title)
    if not m:
        return "UNSPECIFIED"
    return f"HPV{m.group(1)}"


def collapse_to_categories(strain: str) -> str:
    """Map specific strains to the prevalence categories.

    HPV59 → OTHER_HR_HPV; HPV6 → LOW_RISK_HPV; etc.
    """
    if strain in PUBLISHED_PREVALENCE:
        return strain

    # Strip the 'HPV' prefix to get the type number.
    m = re.match(r"HPV(\d+)$", strain)
    if not m:
        return "UNSPECIFIED"
    n = int(m.group(1))

    # IARC Group 1 + 2A high-risk types
    HIGH_RISK = {35, 39, 51, 56, 59, 68, 73, 82, 26, 53, 66, 67, 70}
    LOW_RISK = {6, 11, 40, 42, 43, 44, 54, 61, 72, 81}

    if n in HIGH_RISK:
        return "OTHER_HR_HPV"
    if n in LOW_RISK:
        return "LOW_RISK_HPV"
    # Unrecognized number — treat conservatively as OTHER_HR_HPV.
    return "OTHER_HR_HPV"


def sample_strain(rng: np.random.Generator | None = None) -> str:
    """Sample a strain from the published global prevalence distribution.

    Used for HPV-positive UCI patients where the strain is unknown.
    Tagged SAMPLED_FROM_PRIOR in the provenance system.
    """
    if rng is None:
        rng = np.random.default_rng()
    strains = list(PUBLISHED_PREVALENCE.keys())
    probs = list(PUBLISHED_PREVALENCE.values())
    return rng.choice(strains, p=probs)


def lookup_carcinogenicity(strain: str) -> float:
    """Deterministic carcinogenicity lookup. No sampling, no randomness."""
    return CARCINOGENICITY.get(strain, 0.0)


def live_prevalence_from_ncbi(ncbi_df: pd.DataFrame) -> dict[str, float]:
    """Compute observed strain prevalence in a recent NCBI batch.

    NOT used for sampling — used only for drift detection comparison
    against PUBLISHED_PREVALENCE. NCBI submissions are biased toward
    research interest (HPV16 in particular gets oversampled), so the
    live numbers should NOT replace the published prior.
    """
    if ncbi_df.empty:
        return {}
    strains = ncbi_df["title"].apply(extract_strain_from_title).apply(collapse_to_categories)
    # Drop UNSPECIFIED — we can't compare it to the published prior.
    typed = strains[strains != "UNSPECIFIED"]
    if typed.empty:
        return {}
    counts = Counter(typed)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def report_prior_vs_live(ncbi_df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison of published prior vs current NCBI prevalence.

    Returns a DataFrame ready for printing or logging. Useful for the
    README and the drift module.
    """
    live = live_prevalence_from_ncbi(ncbi_df)
    rows = []
    for strain, prior_p in PUBLISHED_PREVALENCE.items():
        rows.append({
            "strain": strain,
            "published_prior": prior_p,
            "ncbi_live": live.get(strain, 0.0),
            "delta": live.get(strain, 0.0) - prior_p,
        })
    return pd.DataFrame(rows).sort_values("published_prior", ascending=False).reset_index(drop=True)


def main() -> None:
    """Print the published prior and the live NCBI comparison."""
    import sys
    from src.storage.paths import RAW_DIR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    ncbi_files = sorted(RAW_DIR.glob("ncbi_hpv_sequences_*.parquet"))
    if not ncbi_files:
        logger.error("No NCBI ingestion files found in %s; run ingestion first", RAW_DIR)
        sys.exit(1)
    latest = ncbi_files[-1]
    logger.info("Using NCBI batch: %s", latest.name)
    ncbi = pd.read_parquet(latest)
    table = report_prior_vs_live(ncbi)
    print()
    print(table.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
