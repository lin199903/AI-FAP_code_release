# AI-FAP Analysis Code Release

Reproducibility bundle for the AI-FAP study:

> Uncertainty-aware admission risk typing for metabolically driven acute
> pancreatitis under an event-timing, leakage, and transportability audit.

This bundle contains analysis code and documentation only. It contains no raw
PhysioNet data, no record-level release data, and no database credentials.

## Data Access

The analyses require credentialed access to the following databases through
PhysioNet or the relevant data provider:

| Dataset | Role | Access |
|---|---|---|
| MIMIC-IV | Development cohort and internal audit | Credentialed DUA |
| eICU-CRD | Level B transportability audit | Credentialed DUA |
| NWICU | Level B transportability audit | Credentialed DUA |

MIMIC-IV is expected in PostgreSQL with the standard schema-qualified tables
(`mimiciv_hosp.*`, `mimiciv_icu.*`). eICU-CRD and NWICU are read from local
PhysioNet CSV exports via `duckdb`.

## Repository Layout

The release is structured so that this directory itself can be uploaded as a
GitHub repository or archived to Zenodo:

| Path | Role |
|---|---|
| `README.md` | Root documentation for public upload |
| `requirements.txt` | Root dependency file |
| `run_pipeline.py` | Root wrapper entry point |
| `JBI_CODE_UPLOAD_NOTES.md` | Short JBI-facing upload note |
| `04B_FAP_AI/` | Main analysis scripts and detailed release manifest |

## Environment

Use Python 3.11 or later.

```bash
python -m pip install -r requirements.txt
```

The current audit was run with Python 3.13 and the package versions reported in
`04B_FAP_AI/RELEASE_MANIFEST.md`. The `requirements.txt` file uses minimum versions rather
than exact locks to keep the bundle portable.

## Configuration

MIMIC-IV PostgreSQL access is resolved by `04B_FAP_AI/_dbconfig.py` in this order:

1. `PG_DSN` environment variable.
2. A `PG_DSN=...` line in the file named by `MDAP_ENV_FILE`.
3. A local `04B_FAP_AI/.env.local` file.
4. If none is found, the scripts stop with setup instructions.

External audit directories are configured separately:

```bash
# eICU-CRD 2.0 directory containing patient.csv.gz, diagnosis.csv.gz, lab.csv.gz
export EICU_DATA_DIR=/path/to/eicu-crd/2.0

# NWICU data directory containing nw_hosp/ and nw_icu/
export NWICU_DATA_DIR=/path/to/nwicu-northwestern-icu/0.1.0/data
```

On Windows PowerShell:

```powershell
$env:PG_DSN = "postgresql://USER:PASSWORD@HOST:5432/mimiciv"
$env:EICU_DATA_DIR = "D:\physionet\eicu-crd\2.0"
$env:NWICU_DATA_DIR = "D:\physionet\nwicu-northwestern-icu\0.1.0\data"
```

No host, username, password, or local data path is stored in source files.

## Run Order

Run from the repository root.

```bash
python run_pipeline.py --stage internal
python run_pipeline.py --stage external
python run_pipeline.py --stage figures
```

`--stage internal` builds the MIMIC-IV cohort, trajectory audit, landmark
models, robust OOF model, governance outputs, risk typing, negative control, and
revision sensitivity outputs.

`--stage external` runs eICU and NWICU Level B transportability audits and then
regenerates the merged transportability table. It requires both external data
environment variables.

`--stage figures` regenerates manuscript figures from the current output CSVs.

For a one-command local run when all data are configured:

```bash
python run_pipeline.py --stage all
```

## Main Scripts

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_mdap_cohort_build.py` | Build canonical MDAP cohort and Table 1 outputs |
| 2 | `02_mdap_gbtm.py` | 0-48 h trajectory phenotyping and landmark feature exports |
| 3 | `03_landmark_ml.py` | T0/T24/T48 landmark models and leakage-aware performance |
| 4 | `06_robust_cv_oof.py` | Primary repeated-CV OOF model and permutation audit |
| 5 | `06b_oof_outputs.py` | Pooled OOF predictions, calibration, and abstention exports |
| 6 | `04_ai_governance.py` | Calibration, missingness, shortcut, and abstention governance |
| 7 | `08_risk_typing_mapping.py` | Six-type surveillance-priority mapping |
| 8 | `05_eicu_transportability.py` | eICU Level B transportability audit |
| 9 | `06_nwicu_transportability.py` | NWICU Level B transportability audit |
| 10 | `07_table_t2_merged.py` | Merged transportability table from current audit CSVs |
| 11 | `make_figures.py`, `make_workflow_figure.py` | Manuscript figures |

## Key Output Files

Important generated outputs under `04B_FAP_AI/outputs/` include:

| Output | Description |
|---|---|
| `canonical_mdap_cohort.csv` | Derived analysis cohort; do not redistribute without DUA review |
| `table1_summary.csv` | Aggregate Table 1 summary |
| `robust_cv_oof_summary.csv` | Primary OOF model performance |
| `oof_predictions_corrected.csv` | Patient-level OOF predictions; do not redistribute |
| `eicu_transportability_results.csv` | Aggregate eICU audit metrics |
| `eicu_tertile_enrichment.csv` | Aggregate eICU tertile event rates |
| `nwicu_transportability_results.csv` | Aggregate NWICU audit metrics |
| `nwicu_tertile_enrichment.csv` | Aggregate NWICU tertile event rates |
| `table_t2_transportability_merged.csv` | Merged transportability table |

## Redistribution Rules

Do not publish:

- raw PhysioNet tables;
- database credentials or DSNs;
- record-level derived files, including canonical cohorts, prediction rows,
  trajectory assignments, or filtered external cohorts;
- local logs containing absolute paths.

The detailed release manifest is in `04B_FAP_AI/RELEASE_MANIFEST.md`. It
identifies source files that are safe to include in a public code repository
and output files that should be regenerated locally.
