# -*- coding: utf-8 -*-
"""Generate Table 2 from current transportability audit outputs.

This script intentionally avoids hard-coded eICU/NWICU performance values.
Upstream audit scripts must write:
  - eicu_transportability_results.csv
  - eicu_tertile_enrichment.csv
  - eicu_transportability_shift.csv
  - nwicu_transportability_results.csv
  - nwicu_tertile_enrichment.csv
  - nwicu_transportability_shift.csv
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np


OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

MIMIC_OOF = {
    "AUROC": 0.735,
    "Brier": 0.173,
    "Cal_intercept": 0.000,
    "Cal_slope": 0.850,
    "N": 483,
    "N_events": 149,
    "N_features": 21,
}


def read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(OUTDIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required audit output: {path}")
    return pd.read_csv(path)


def metric(df: pd.DataFrame, name: str, default=np.nan):
    row = df.loc[df["metric"] == name, "value"]
    if len(row) == 0:
        return default
    value = row.iloc[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def fmt_num(value, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def rank_string(results: pd.DataFrame) -> str:
    r = metric(results, "Spearman_r")
    p = metric(results, "Spearman_p")
    if pd.isna(r):
        return "--"
    if pd.isna(p):
        return fmt_num(r)
    return f"{r:.3f} (p={p:.3f})"


def tertile_string(tertiles: pd.DataFrame) -> str:
    rates = tertiles.set_index("tertile")
    if "High" not in rates.index or "Low" not in rates.index:
        return "--"
    high = rates.loc["High", "event_rate_pct"]
    low = rates.loc["Low", "event_rate_pct"]
    return f"{high:.1f}% (vs {low:.1f}% low)"


def shifted_string(shift: pd.DataFrame) -> str:
    return f"{int((shift['shift_flag'] == 'YES').sum())}/{len(shift)}"


def epv(events: float, n_features: int) -> str:
    return f"{events / n_features:.1f}" if n_features else "--"


def row(metric_name: str, mimic, eicu, nwicu, note: str) -> dict:
    return {
        "metric": metric_name,
        "MIMIC-IV (dev)": mimic,
        "eICU-CRD (audit)": eicu,
        "NWICU (audit)": nwicu,
        "note": note,
    }


eicu = read_csv("eicu_transportability_results.csv")
nwicu = read_csv("nwicu_transportability_results.csv")
eicu_shift = read_csv("eicu_transportability_shift.csv")
nwicu_shift = read_csv("nwicu_transportability_shift.csv")
eicu_tertiles = read_csv("eicu_tertile_enrichment.csv")
nwicu_tertiles = read_csv("nwicu_tertile_enrichment.csv")

eicu_n = int(metric(eicu, "N"))
eicu_events = int(metric(eicu, "N_events"))
nwicu_n = int(metric(nwicu, "N"))
nwicu_events = int(metric(nwicu, "N_events"))

rows = [
    row(
        "N (admissions)",
        MIMIC_OOF["N"],
        eicu_n,
        nwicu_n,
        "Two small-sample directional audits; not Level A validation.",
    ),
    row(
        "Composite events",
        MIMIC_OOF["N_events"],
        eicu_events,
        nwicu_events,
        "Endpoint structure differs across audit databases; interpret directionally.",
    ),
    row(
        "Events per feature (EPV)",
        epv(MIMIC_OOF["N_events"], MIMIC_OOF["N_features"]),
        epv(eicu_events, len(eicu_shift)),
        epv(nwicu_events, len(nwicu_shift)),
        "External audits remain below conventional validation sample-size targets.",
    ),
    row(
        "AUROC (locked admission model)",
        fmt_num(MIMIC_OOF["AUROC"]),
        fmt_num(metric(eicu, "AUROC")),
        fmt_num(metric(nwicu, "AUROC")),
        "Risk ranking signal is directional and sample-size limited.",
    ),
    row(
        "Brier score",
        fmt_num(MIMIC_OOF["Brier"]),
        fmt_num(metric(eicu, "Brier")),
        fmt_num(metric(nwicu, "Brier")),
        "Absolute risk is not transported without local recalibration.",
    ),
    row(
        "Calibration intercept (ideal=0)",
        fmt_num(MIMIC_OOF["Cal_intercept"]),
        fmt_num(metric(eicu, "Cal_intercept")),
        fmt_num(metric(nwicu, "Cal_intercept")),
        "External calibration-in-the-large is shifted.",
    ),
    row(
        "Calibration slope (ideal=1)",
        fmt_num(MIMIC_OOF["Cal_slope"]),
        fmt_num(metric(eicu, "Cal_slope")),
        fmt_num(metric(nwicu, "Cal_slope")),
        "External calibration slopes deviate from 1; local recalibration is required.",
    ),
    row(
        "Spearman rank rho",
        "--",
        rank_string(eicu),
        rank_string(nwicu),
        "Rank preservation is exploratory because both audits are small.",
    ),
    row(
        "High-risk tertile event rate",
        "--",
        tertile_string(eicu_tertiles),
        tertile_string(nwicu_tertiles),
        "High-tertile enrichment is descriptive, not a deployment threshold.",
    ),
    row(
        "Features with SMD>0.2",
        "--",
        shifted_string(eicu_shift),
        shifted_string(nwicu_shift),
        "Feature/coding shift is a headline transportability finding.",
    ),
    row(
        "Evaluation framing",
        "Development (OOF)",
        "Level B transportability audit",
        "Level B transportability audit",
        "Neither audit constitutes definitive external validation.",
    ),
]

df_table2 = pd.DataFrame(rows)
csv_path = os.path.join(OUTDIR, "table_t2_transportability_merged.csv")
df_table2.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
print(df_table2.to_string(index=False))

df_shift_merged = eicu_shift.merge(
    nwicu_shift[["feature", "smd", "ks_stat", "ks_p", "shift_flag"]].rename(
        columns={
            "smd": "smd_nwicu",
            "ks_stat": "ks_stat_nwicu",
            "ks_p": "ks_p_nwicu",
            "shift_flag": "shift_flag_nwicu",
        }
    ),
    on="feature",
    how="outer",
).rename(
    columns={
        "smd": "smd_eicu",
        "ks_stat": "ks_stat_eicu",
        "ks_p": "ks_p_eicu",
        "shift_flag": "shift_flag_eicu",
    }
)
shift_csv = os.path.join(OUTDIR, "table_t2b_feature_shift_merged.csv")
df_shift_merged.to_csv(shift_csv, index=False)
print(f"Saved: {shift_csv}")

latex = r"""\begin{table*}[h]
\centering
\caption{\textbf{Transportability audit across two MIMIC-compatible databases.}
Both eICU-CRD and NWICU are presented as \emph{Level B} directional
transportability audits, not Level A external validation. All external metrics
are generated from the locked admission model and current audit CSV outputs.}
\label{tab:t2}
\footnotesize
\begin{tabular}{lcccl}
\toprule
\textbf{Metric} & \textbf{MIMIC-IV (dev)} & \textbf{eICU-CRD (audit)} & \textbf{NWICU (audit)} & \textbf{Interpretation} \\
\midrule
"""
for r in rows:
    latex += (
        f"{r['metric']} & {r['MIMIC-IV (dev)']} & "
        f"{r['eICU-CRD (audit)']} & {r['NWICU (audit)']} & "
        f"{r['note']} \\\\\n"
    )
latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""

tex_path = os.path.join(OUTDIR, "table_t2_transportability_merged.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"Saved: {tex_path}")

shift_latex = r"""\begin{table}[h]
\centering
\caption{\textbf{Feature distribution shift using MIMIC-IV as reference.}
Standardized mean difference (SMD) and Kolmogorov-Smirnov statistics are
reported for features available in the corresponding audit database. Asterisks
mark SMD$>$0.2.}
\label{tab:t2b}
\scriptsize
\begin{tabular}{lcccc}
\toprule
\textbf{Feature} & \textbf{SMD eICU} & \textbf{SMD NWICU} & \textbf{KS eICU} & \textbf{KS NWICU} \\
\midrule
"""
for feat in sorted(set(eicu_shift["feature"]) | set(nwicu_shift["feature"])):
    er = eicu_shift.loc[eicu_shift["feature"] == feat]
    nr = nwicu_shift.loc[nwicu_shift["feature"] == feat]

    def smd_cell(df: pd.DataFrame) -> str:
        if len(df) == 0:
            return "--"
        suffix = "*" if df["shift_flag"].iloc[0] == "YES" else ""
        return f"{float(df['smd'].iloc[0]):.2f}{suffix}"

    def ks_cell(df: pd.DataFrame) -> str:
        if len(df) == 0:
            return "--"
        return f"{float(df['ks_stat'].iloc[0]):.2f}"

    shift_latex += (
        f"{feat} & {smd_cell(er)} & {smd_cell(nr)} & "
        f"{ks_cell(er)} & {ks_cell(nr)} \\\\\n"
    )
shift_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

shift_tex_path = os.path.join(OUTDIR, "table_t2b_feature_shift_merged.tex")
with open(shift_tex_path, "w", encoding="utf-8") as f:
    f.write(shift_latex)
print(f"Saved: {shift_tex_path}")
