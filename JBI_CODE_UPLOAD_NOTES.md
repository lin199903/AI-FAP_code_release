# JBI Code Upload Notes

This directory is the public code package prepared for the JBI submission of the
AI-FAP study.

## What this package is for

- GitHub repository upload
- Zenodo DOI archive, if a DOI-backed code citation is preferred
- Reviewer-facing reproducibility package after credentialed data access

## What is included

- Source code only
- Environment requirements
- Pipeline entry point
- Release README and manifest

## What is intentionally excluded

- Raw MIMIC-IV, eICU-CRD, and NWICU data
- Database credentials and DSNs
- Patient-level derived cohorts or prediction rows
- Local logs and absolute local paths

## Recommended repository root

Use this folder itself as the repository root:

- `README.md`
- `requirements.txt`
- `run_pipeline.py`
- `04B_FAP_AI/`

## Suggested manuscript wording

Code availability:
Analysis code for the AI-FAP study is provided in a public repository / DOI
archive. The release contains source code and documentation only; credentialed
access to MIMIC-IV, eICU-CRD, and NWICU is required to regenerate derived
datasets and model outputs.
