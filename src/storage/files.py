"""Helpers for friendly file-missing errors.

Every loader in the pipeline can hit FileNotFoundError when its prerequisite
data hasn't been produced yet. Instead of a raw traceback, use these helpers
to surface a clear message that tells the user which command to run next.
"""
from __future__ import annotations

import sys
from pathlib import Path


def require_file(
    path: Path,
    hint_command: str,
    description: str | None = None,
) -> None:
    """Ensure a file exists; if not, print a friendly error and exit cleanly.

    Parameters
    ----------
    path : Path
        The file expected to exist.
    hint_command : str
        The command the user should run to produce this file.
    description : str | None
        Optional one-line description of what this file is, shown in the
        error message for extra clarity.
    """
    if path.exists():
        return

    print(file=sys.stderr)
    print(f"ERROR: Required file not found.", file=sys.stderr)
    print(f"  Expected at: {path}", file=sys.stderr)
    if description:
        print(f"  Description: {description}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"To create it, run:", file=sys.stderr)
    print(f"  {hint_command}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"Or for a one-command full setup:", file=sys.stderr)
    print(f"  .\\run.ps1 quickstart", file=sys.stderr)
    print(file=sys.stderr)
    sys.exit(2)
