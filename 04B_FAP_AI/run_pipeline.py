# -*- coding: utf-8 -*-
"""Run the AI-FAP analysis pipeline in reproducible stages."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent

INTERNAL_SCRIPTS = [
    "01_mdap_cohort_build.py",
    "_deterioration_timing.py",
    "02_mdap_gbtm.py",
    "03_landmark_ml.py",
    "06_robust_cv_oof.py",
    "06b_oof_outputs.py",
    "04_ai_governance.py",
    "08_risk_typing_mapping.py",
    "07_sepsis_contrast.py",
    "analysis_M3_M5_baselines.py",
    "analysis_Q4_gbtm_stability.py",
]

EXTERNAL_SCRIPTS = [
    "05_eicu_transportability.py",
    "06_nwicu_transportability.py",
    "07_table_t2_merged.py",
]

FIGURE_SCRIPTS = [
    "make_figures.py",
    "make_workflow_figure.py",
    "make_graphical_abstract.py",
]


def require_env(var_name: str) -> None:
    if not os.getenv(var_name):
        raise RuntimeError(f"Set {var_name} before running this stage.")


def run_script(script_name: str) -> None:
    print(f"\n=== Running {script_name} ===", flush=True)
    subprocess.run([sys.executable, script_name], cwd=BASE, check=True)


def run_many(script_names: list[str]) -> None:
    for script_name in script_names:
        run_script(script_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["internal", "external", "figures", "all"],
        default="internal",
        help="Pipeline stage to run. Default: internal.",
    )
    args = parser.parse_args()

    if args.stage in {"internal", "all"}:
        run_many(INTERNAL_SCRIPTS)

    if args.stage in {"external", "all"}:
        require_env("EICU_DATA_DIR")
        require_env("NWICU_DATA_DIR")
        run_many(EXTERNAL_SCRIPTS)

    if args.stage in {"figures", "all"}:
        run_many(FIGURE_SCRIPTS)

    print("\nPipeline stage completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
