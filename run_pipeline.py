"""Top-level wrapper for the AI-FAP public code-release pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent
INNER = BASE / "04B_FAP_AI" / "run_pipeline.py"


def main() -> int:
    cmd = [sys.executable, str(INNER), *sys.argv[1:]]
    return subprocess.call(cmd, cwd=BASE / "04B_FAP_AI")


if __name__ == "__main__":
    raise SystemExit(main())
