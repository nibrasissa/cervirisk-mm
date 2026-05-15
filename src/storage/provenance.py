"""Provenance tracking for CerviRisk-MM.

The core integrity rule of this project: every data point must carry a label
indicating whether it is real-observed, real-derived, or sampled/constructed.
Real outcomes are never overwritten; constructed combinations are never
confused with real observations.

This module defines the enum and helper functions that enforce that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provenance(str, Enum):
    """Where a value originated, and whether it reflects observation."""

    REAL_OBSERVED = "REAL_OBSERVED"
    """Directly observed in a real source (e.g. UCI age, biopsy result)."""

    REAL_OUTCOME = "REAL_OUTCOME"
    """A real ground-truth outcome. Never modified by augmentation."""

    REAL_LOOKUP = "REAL_LOOKUP"
    """Deterministic lookup in a curated reference table (e.g. PaVE)."""

    REAL_COMPUTED = "REAL_COMPUTED"
    """Computed from real inputs via a published algorithm (e.g. PRS)."""

    SAMPLED_FROM_PRIOR = "SAMPLED_FROM_PRIOR"
    """Sampled from a published distribution (e.g. HPV strain prevalence)."""

    CONSTRUCTED = "CONSTRUCTED"
    """A combination of unrelated real components into one record."""


@dataclass
class ProvenanceTag:
    """Travels with each augmented feature so origin is never lost."""

    provenance: Provenance
    source: str
    citation: str = ""
    notes: str = ""


@dataclass
class PatientRecord:
    """A patient record with per-feature provenance.

    The biopsy outcome is the only field protected from augmentation; any
    code that modifies it will fail the immutability test in tests/.
    """

    patient_id: str
    features: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ProvenanceTag] = field(default_factory=dict)
    biopsy_outcome: int | None = None  # 0/1, set once from UCI, then frozen

    def set_feature(
        self,
        name: str,
        value: Any,
        tag: ProvenanceTag,
    ) -> None:
        if name == "biopsy_outcome":
            raise ValueError(
                "Biopsy outcome is protected. Use anchor_from_uci() once at "
                "patient creation; never modify thereafter."
            )
        self.features[name] = value
        self.provenance[name] = tag

    def has_synthetic_components(self) -> bool:
        return any(
            tag.provenance in (Provenance.SAMPLED_FROM_PRIOR, Provenance.CONSTRUCTED)
            for tag in self.provenance.values()
        )

    def data_status(self) -> str:
        """Single-string summary suitable for API responses."""
        if self.has_synthetic_components():
            return "SYNTHETIC_ASSEMBLY"
        return "REAL_ONLY"
