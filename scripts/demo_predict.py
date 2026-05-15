"""End-to-end prediction demo for CerviRisk-MM.

Loads the trained model from disk and runs a sample patient through the
pipeline, narrating each stage. Useful for:
  - Reviewers who want to see what the API does without curl
  - Interview demos
  - Sanity-checking after retraining

Run with:  python -m scripts.demo_predict   (or:  .\run.ps1 predict)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.storage.paths import MODELS_DIR


# ANSI colors for the narration
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---- Sample patients --------------------------------------------------------
# Three realistic cases representing the spectrum the triage model handles.
SAMPLE_PATIENTS = [
    {
        "label": "Low-risk profile",
        "description": "young, no smoking, no HPV evidence, normal screening",
        "features": {
            "Age": 24,
            "Number of sexual partners": 1,
            "First sexual intercourse": 19,
            "Num of pregnancies": 0,
            "Smokes": 0,
            "Smokes (years)": 0,
            "Hormonal Contraceptives": 0,
            "IUD": 0,
            "STDs": 0,
            "STDs:HPV": 0,
            "Dx:HPV": 0,
            "Hinselmann": 0,
            "Schiller": 0,
            "Citology": 0,
        },
    },
    {
        "label": "Moderate-risk profile",
        "description": "established risk factors, HPV positive, one screening flag",
        "features": {
            "Age": 38,
            "Number of sexual partners": 4,
            "First sexual intercourse": 16,
            "Num of pregnancies": 2,
            "Smokes": 1,
            "Smokes (years)": 12,
            "Hormonal Contraceptives": 1,
            "Hormonal Contraceptives (years)": 6,
            "IUD": 0,
            "STDs": 1,
            "STDs:HPV": 1,
            "Dx:HPV": 1,
            "Hinselmann": 0,
            "Schiller": 1,
            "Citology": 0,
        },
    },
    {
        "label": "High-risk profile",
        "description": "all three screening tests positive, HPV history",
        "features": {
            "Age": 45,
            "Number of sexual partners": 6,
            "First sexual intercourse": 15,
            "Num of pregnancies": 3,
            "Smokes": 1,
            "Smokes (years)": 25,
            "Hormonal Contraceptives": 1,
            "Hormonal Contraceptives (years)": 15,
            "IUD": 1,
            "IUD (years)": 5,
            "STDs": 1,
            "STDs (number)": 2,
            "STDs:HPV": 1,
            "Dx:HPV": 1,
            "Dx:CIN": 1,
            "Hinselmann": 1,
            "Schiller": 1,
            "Citology": 1,
        },
    },
]


# ---- Helpers ----------------------------------------------------------------
def _tier(p: float) -> tuple[str, str]:
    """Return (tier_label, color)."""
    if p < 0.20:
        return "LOW", GREEN
    if p < 0.50:
        return "MODERATE", YELLOW
    return "HIGH", RED


def _load_model_and_metadata():
    model_path = MODELS_DIR / "cervirisk_mm_v0.1.pkl"
    if not model_path.exists():
        print(f"\n{RED}ERROR:{RESET} Model artifact not found at {model_path}")
        print()
        print("This demo needs a trained model. Run one of:")
        print(f"  .\\run.ps1 fit             quick training  (~30 sec)")
        print(f"  .\\run.ps1 train           full tuning     (~60 min)")
        print()
        sys.exit(1)

    model = joblib.load(model_path)
    best_variant = None
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            best = max(metrics, key=lambda r: ((r.get("dev") or {})
                        .get("youden", {}).get("auprc", 0)
                        if isinstance(r.get("dev"), dict) else 0))
            best_variant = f"{best.get('mode')} + {best.get('model')}"
        except Exception:
            pass
    return model, best_variant


# ---- Main demo --------------------------------------------------------------
def main() -> None:
    print()
    print("=" * 70)
    print(f"  {BOLD}CerviRisk-MM end-to-end prediction demo{RESET}")
    print("=" * 70)

    # Stage 1: load
    print(f"\n[1/4] Loading trained model from disk...")
    model, best_variant = _load_model_and_metadata()
    print(f"      {GREEN}OK{RESET}  loaded "
          f"{DIM}({best_variant or 'unknown variant'}){RESET}")

    # Run all three sample patients
    for i, sample in enumerate(SAMPLE_PATIENTS, 1):
        print()
        print(f"{CYAN}--- Patient {i}/{len(SAMPLE_PATIENTS)}: "
              f"{sample['label']} ---{RESET}")
        print(f"{DIM}    {sample['description']}{RESET}\n")

        feats = sample["features"]
        df = pd.DataFrame([feats])

        # Stage 2: preprocess (handled by the pipeline)
        print(f"[2/4] Preprocessing features...")
        print(f"      {GREEN}OK{RESET}  {len(feats)} features in, "
              f"NaN-imputation + standardization handled by ColumnTransformer")

        # Stage 3: predict
        print(f"[3/4] Running prediction...")
        try:
            proba = float(model.predict_proba(df)[0, 1])
        except Exception as e:
            print(f"      {RED}FAIL{RESET}  {type(e).__name__}: {e}")
            continue
        print(f"      {GREEN}OK{RESET}  risk_probability = {proba:.4f}")

        # Stage 4: tier
        print(f"[4/4] Mapping probability to clinical tier...")
        tier, color = _tier(proba)
        print(f"      {GREEN}OK{RESET}  tier = {color}{tier}{RESET}")

        # Brief patient summary
        print(f"\n  Risk score:       {color}{proba*100:5.1f}%{RESET}")
        print(f"  Clinical tier:    {color}{tier}{RESET}")
        recommendation = {
            "LOW": "routine screening, next cycle",
            "MODERATE": "expedited follow-up; repeat cytology in 6 months",
            "HIGH": "refer for diagnostic biopsy",
        }[tier]
        print(f"  Recommendation:   {recommendation}")

    # Footer
    print()
    print("-" * 70)
    print(f"{DIM}  Generated at {datetime.now(timezone.utc).isoformat(timespec='seconds')}{RESET}")
    print(f"{DIM}  Research prototype — not a medical device.{RESET}")
    print(f"{DIM}  Full disclaimer: see /predict/cervical-risk API response.{RESET}")
    print()


if __name__ == "__main__":
    main()
