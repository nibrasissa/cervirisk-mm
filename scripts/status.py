"""Show the current state of the CerviRisk-MM pipeline.

Checks which files exist, when they were modified, and what the
recommended next action is. Useful for picking up the project after
days/weeks away, or for a reviewer who wants a quick sense of state.

Run with:  python -m scripts.status     (or:  .\run.ps1 status)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.storage.paths import (
    DATA_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    genomes_1kg_panel_path,
    uci_raw_path,
)


# ANSI color helpers (Windows PowerShell 5.1+ supports these)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _file_info(path: Path) -> dict:
    """Return existence, size, age, and row count (parquet only) for a file."""
    if not path.exists():
        return {"ok": False}
    stat = path.stat()
    info: dict = {
        "ok": True,
        "size_kb": round(stat.st_size / 1024, 1),
        "mtime": datetime.fromtimestamp(stat.st_mtime),
    }
    if path.suffix == ".parquet":
        try:
            import pandas as pd
            info["rows"] = len(pd.read_parquet(path))
        except Exception:
            pass
    return info


def _fmt_age(dt: datetime) -> str:
    delta = datetime.now() - dt
    if delta.days > 30:
        return f"{delta.days // 30} months ago"
    if delta.days > 0:
        return f"{delta.days} days ago"
    if delta.seconds > 3600:
        return f"{delta.seconds // 3600} hours ago"
    if delta.seconds > 60:
        return f"{delta.seconds // 60} minutes ago"
    return "just now"


def _check(name: str, info: dict, extra: str = "") -> tuple[bool, str]:
    if info.get("ok"):
        bits = []
        if "rows" in info:
            bits.append(f"{info['rows']} rows")
        bits.append(_fmt_age(info["mtime"]))
        detail = ", ".join(bits)
        line = f"  {GREEN}OK{RESET}      {name:<28} {detail}"
        if extra:
            line += f" {DIM}({extra}){RESET}"
        return True, line
    return False, f"  {RED}MISSING{RESET} {name}"


def main() -> None:
    print()
    print(f"{BOLD}CerviRisk-MM project status{RESET}")
    print("=" * 60)

    # ---- Data files --------------------------------------------------------
    print(f"\n{YELLOW}Data:{RESET}")
    data_states = []

    ok, line = _check("UCI parquet", _file_info(uci_raw_path()))
    print(line); data_states.append(("uci", ok))

    ok, line = _check("1000G panel", _file_info(genomes_1kg_panel_path()))
    print(line); data_states.append(("1kg", ok))

    ncbi_files = sorted(RAW_DIR.glob("ncbi_hpv_sequences_*.parquet"))
    if ncbi_files:
        ok, line = _check("NCBI HPV sequences",
                           _file_info(ncbi_files[-1]),
                           extra=f"latest: {ncbi_files[-1].name}")
        print(line); data_states.append(("ncbi", ok))
    else:
        print(f"  {RED}MISSING{RESET} NCBI HPV sequences")
        data_states.append(("ncbi", False))

    assembled = PROCESSED_DIR / "patients_assembled.parquet"
    ok_assembled, line = _check("Assembled patients", _file_info(assembled))
    print(line); data_states.append(("assembled", ok_assembled))

    # ---- Model artifacts ---------------------------------------------------
    print(f"\n{YELLOW}Model:{RESET}")
    model_states = []

    model_path = MODELS_DIR / "cervirisk_mm_v0.1.pkl"
    info = _file_info(model_path)
    extra = ""
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            if metrics:
                best = max(metrics, key=lambda r: ((r.get("dev") or {})
                            .get("youden", {}).get("auprc", 0)
                            if isinstance(r.get("dev"), dict) else 0))
                extra = f"{best.get('mode')} + {best.get('model')}"
        except Exception:
            pass
    ok, line = _check("Trained model", info, extra=extra)
    print(line); model_states.append(("model", ok))

    ok, line = _check("Metrics report", _file_info(metrics_path))
    print(line); model_states.append(("metrics", ok))

    baseline_path = MODELS_DIR / "drift_baseline.json"
    info = _file_info(baseline_path)
    extra = ""
    if info.get("ok"):
        try:
            bl = json.loads(baseline_path.read_text())
            extra = f"{bl.get('n_records')} patients"
        except Exception:
            pass
    ok, line = _check("Drift baseline", info, extra=extra)
    print(line); model_states.append(("baseline", ok))

    # ---- API status --------------------------------------------------------
    print(f"\n{YELLOW}API:{RESET}")
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        result = sock.connect_ex(("127.0.0.1", 8000))
        sock.close()
        if result == 0:
            print(f"  {GREEN}RUNNING{RESET} on http://localhost:8000/  "
                  f"{DIM}(visit /docs for Swagger UI){RESET}")
        else:
            print(f"  {DIM}not running.{RESET} Start with: .\\run.ps1 serve")
    except Exception:
        print(f"  {DIM}status check unavailable{RESET}")

    # ---- Test suite --------------------------------------------------------
    print(f"\n{YELLOW}Tests:{RESET}")
    test_files = list(Path("tests").glob("test_*.py"))
    if test_files:
        print(f"  {GREEN}OK{RESET}      {len(test_files)} test files "
              f"{DIM}(run with .\\run.ps1 test){RESET}")
    else:
        print(f"  {RED}MISSING{RESET} no test files found")

    # ---- Recommended next action -------------------------------------------
    print()
    print(f"{BOLD}Recommended next action:{RESET}")
    if not all(s for _, s in data_states[:3]):    # raw data missing
        print(f"  .\\run.ps1 ingest          download the four data sources")
    elif not ok_assembled:
        print(f"  .\\run.ps1 assemble        build the assembled patient table")
    elif not any(s for _, s in model_states[:1]):
        print(f"  .\\run.ps1 fit             quick training (~30 sec)")
        print(f"  .\\run.ps1 train           full tuning  (~60 min)")
    elif not model_states[2][1]:    # baseline missing
        print(f"  .\\run.ps1 baseline        capture drift baseline")
    else:
        print(f"  .\\run.ps1 serve           boot the API on port 8000")
        print(f"  .\\run.ps1 drift           run the drift detection demo")
        print(f"  .\\run.ps1 predict         run the end-to-end prediction demo")
    print()


if __name__ == "__main__":
    main()
