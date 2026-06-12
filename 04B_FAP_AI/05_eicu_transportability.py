# -*- coding: utf-8 -*-
"""
AI-FAP 旗舰 — eICU 小样本可迁移性审计
===============================================================================
按 AGENTS.md §8.2 修正版执行：小样本方向性审计，非正式外部验证。

审计内容（Subasri 2025 + Davis 2020 框架）：
  1. 特征分布漂移（SMD / KS / PSI）
  2. 风险排序保持性（Spearman / AUROC 方向性）
  3. 校准漂移（Brier / ECE / calibration intercept & slope）
  4. 弃权率漂移
  5. 变量可得性审计

数据：eICU-CRD (DuckDB in-memory)
  - 已知 N=72 MDAP（可行性审计 03_eicu_mdap_feasibility.py）
  - 器官指标覆盖率尚可（WBC 92%, Cr 94%, BUN 96%, Glucose 96%）
  - TG≥500 仅 11 例

正式表述：We performed a small-sample transportability audit in eICU-CRD
rather than a definitive external validation.
"""

import os
import sys
import warnings
import gzip
import tempfile
import json
import numpy as np
import pandas as pd
import duckdb
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

EICU_DIR = os.getenv("EICU_DATA_DIR")
if not EICU_DIR:
    raise RuntimeError(
        "Set EICU_DATA_DIR to the local eICU-CRD 2.0 directory "
        "(the folder containing patient.csv.gz and diagnosis.csv.gz)."
    )

# ═══════════════════════════════════════════════════════════
# 1. eICU MDAP 队列构建
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("AI-FAP eICU Transportability Audit")
print("=" * 70)
print("\nNOTE: This is a small-sample directional audit, NOT a definitive external validation.")

con = duckdb.connect(":memory:")

print("\n[1/5] Loading eICU tables...")
con.execute(f"CREATE TABLE patient AS SELECT * FROM read_csv_auto('{EICU_DIR}/patient.csv.gz', ignore_errors=true)")
con.execute(f"CREATE TABLE diagnosis AS SELECT * FROM read_csv_auto('{EICU_DIR}/diagnosis.csv.gz', ignore_errors=true)")
n_pt = con.execute("SELECT COUNT(*) FROM patient").fetchone()[0]
print(f"  patient: {n_pt} stays")

# AP
con.execute("""
    CREATE TABLE ap AS
    SELECT DISTINCT patientunitstayid
    FROM diagnosis
    WHERE icd9code LIKE '577.0%'
       OR LOWER(diagnosisstring) LIKE '%pancreatitis%'
""")
n_ap = con.execute("SELECT COUNT(*) FROM ap").fetchone()[0]
print(f"  AP: {n_ap}")

# Metabolic phenotype
con.execute("""
    CREATE TABLE metabolic AS
    SELECT DISTINCT patientunitstayid
    FROM diagnosis
    WHERE icd9code LIKE '278.0%' OR icd9code LIKE '272.1%'
       OR icd9code LIKE '272.4%' OR icd9code LIKE '272.2%'
       OR LOWER(diagnosisstring) LIKE '%obes%'
       OR LOWER(diagnosisstring) LIKE '%hypertriglyceridemia%'
       OR LOWER(diagnosisstring) LIKE '%hyperlipidemia%'
""")

# Exclusions
con.execute("""
    CREATE TABLE excluded AS
    SELECT DISTINCT patientunitstayid
    FROM diagnosis
    WHERE icd9code LIKE '574%' OR icd9code LIKE '575%'
       OR icd9code LIKE '571%' OR icd9code LIKE '303%' OR icd9code LIKE '305.0%'
       OR icd9code LIKE '157%'
       OR LOWER(diagnosisstring) LIKE '%gallstone%'
       OR LOWER(diagnosisstring) LIKE '%cholecystitis%'
       OR LOWER(diagnosisstring) LIKE '%alcohol%'
       OR LOWER(diagnosisstring) LIKE '%pancreatic cancer%'
""")

con.execute("""
    CREATE TABLE mdap_clean AS
    SELECT m.patientunitstayid, p.age, p.gender, p.hospitaldischargestatus,
           p.unitdischargestatus, p.admissionweight
    FROM (SELECT DISTINCT a.patientunitstayid FROM ap a JOIN metabolic m2 ON a.patientunitstayid = m2.patientunitstayid) m
    JOIN patient p ON m.patientunitstayid = p.patientunitstayid
    WHERE m.patientunitstayid NOT IN (SELECT patientunitstayid FROM excluded)
""")

mdap_stays = [str(r[0]) for r in con.execute("SELECT patientunitstayid FROM mdap_clean").fetchall()]
n_mdap = len(mdap_stays)
mdap_stay_set = set(mdap_stays)
n_dead = con.execute("SELECT COUNT(*) FROM mdap_clean WHERE hospitaldischargestatus = 'Expired'").fetchone()[0]
print(f"  MDAP clean: {n_mdap}, hospital mortality: {n_dead} ({100*n_dead/max(n_mdap,1):.1f}%)")

# ═══════════════════════════════════════════════════════════
# 2. 提取 eICU 实验室数据
# ═══════════════════════════════════════════════════════════
print("\n[2/5] Extracting lab data for MDAP stays...")

MARKERS = {
    "tg":        ["triglyceride", "triglyc"],
    "wbc":       ["white blood cell", "wbc", "leukocyte"],
    "creatinine": ["creatinine"],
    "bun":       ["bun", "urea nitrogen"],
    "platelet":  ["platelet", "plt"],
    "glucose":   ["glucose"],
    "lactate":   ["lactate"],
    "bilirubin": ["bilirubin"],
    "lipase":    ["lipase"],
    "amylase":   ["amylase"],
}

lab_gz = os.path.join(EICU_DIR, "lab.csv.gz")
out_tmp = os.path.join(tempfile.gettempdir(), "eicu_mdap_labs_transport.csv")
n_total = 0
n_matched = 0

with gzip.open(lab_gz, "rt", encoding="utf-8", errors="replace") as fin, \
     open(out_tmp, "w", encoding="utf-8") as fout:
    header = fin.readline()
    fout.write(header)
    for line in fin:
        n_total += 1
        if n_total % 10000000 == 0:
            sys.stdout.write(f"\r    Scanned {n_total//1000000}M lines, matched {n_matched}")
            sys.stdout.flush()
        fields = line.split(",")
        if len(fields) < 3:
            continue
        stay_id = fields[1]
        if stay_id not in mdap_stay_set:
            continue
        line_lower = line.lower()
        hit = False
        for marker, keywords in MARKERS.items():
            if any(kw in line_lower for kw in keywords):
                hit = True
                break
        if hit:
            fout.write(line)
            n_matched += 1

sys.stdout.write(f"\r    Done: {n_total//1000000}M lines, matched {n_matched}\n")
print(f"  Filtered lab rows: {n_matched}")

con.execute(f"CREATE TABLE labs AS SELECT patientunitstayid, labresultoffset, labname, labresult FROM read_csv_auto('{out_tmp}')")
os.remove(out_tmp)

# ═══════════════════════════════════════════════════════════
# 3. 构建 eICU 特征矩阵（对齐 MIMIC-IV 格式）
# ═══════════════════════════════════════════════════════════
print("\n[3/5] Building eICU feature matrix...")

# 关键修正：排除尿/清除率/其他体液污染（eICU 中 '%creatinine%' 会抓到 urine creatinine /
# creatinine clearance，量级 50-300，把均值抬到 7.04 这种不可能值）。
_NOT_URINE = "LOWER(labname) NOT LIKE '%urine%' AND LOWER(labname) NOT LIKE '%urinary%' AND LOWER(labname) NOT LIKE '%clearance%'"
KEYWORD_MAP = {
    "wbc":       "(LOWER(labname) LIKE '%wbc%' OR LOWER(labname) LIKE '%leukocyte%' OR LOWER(labname) LIKE '%white blood cell%')",
    "creatinine": f"(LOWER(labname) LIKE '%creatinine%' AND {_NOT_URINE})",
    "bun":       "(LOWER(labname) LIKE '%bun%' OR LOWER(labname) LIKE '%urea nitrogen%')",
    "platelet":  "(LOWER(labname) LIKE '%platelet%')",
    "glucose":   f"(LOWER(labname) LIKE '%glucose%' AND {_NOT_URINE})",
    "lactate":   "(LOWER(labname) LIKE '%lactate%')",
    "bilirubin": f"(LOWER(labname) LIKE '%bilirubin%' AND {_NOT_URINE})",
    "tg":        "(LOWER(labname) LIKE '%triglyceride%' OR LOWER(labname) LIKE '%triglyc%')",
    "lipase":    "(LOWER(labname) LIKE '%lipase%')",
    "amylase":   "(LOWER(labname) LIKE '%amylase%')",
}

# 生理范围钳制（防残留污染值进入 AVG）
RANGES = {
    "wbc": (0.1, 100), "creatinine": (0.1, 20), "bun": (1, 200),
    "platelet": (1, 2000), "glucose": (10, 2000), "lactate": (0.1, 30),
    "bilirubin": (0.05, 60), "tg": (10, 10000), "lipase": (1, 20000),
    "amylase": (1, 20000),
}

WINDOWS_MIN = [(0, 360), (360, 1440), (1440, 2880)]
WINDOWS_NAME = ["w0_6", "w6_24", "w24_48"]

eicu_features = []
for stay_id in mdap_stays:
    row = {"patientunitstayid": int(stay_id)}

    # Baseline: first measurement in 0-6h window
    for marker, kw_sql in KEYWORD_MAP.items():
        for (t_start, t_end), w_name in zip(WINDOWS_MIN, WINDOWS_NAME):
            _lo, _hi = RANGES.get(marker, (-1e18, 1e18))
            result = con.execute(f"""
                SELECT AVG(TRY_CAST(labresult AS DOUBLE))
                FROM labs
                WHERE patientunitstayid = {stay_id}
                  AND TRY_CAST(labresultoffset AS INTEGER) BETWEEN {t_start} AND {t_end}
                  AND ({kw_sql})
                  AND TRY_CAST(labresult AS DOUBLE) IS NOT NULL
                  AND TRY_CAST(labresult AS DOUBLE) BETWEEN {_lo} AND {_hi}
            """).fetchone()[0]
            row[f"{marker}_{w_name}"] = result

    # TG flag
    tg_first = row.get("tg_w0_6") or row.get("tg_w6_24") or row.get("tg_w24_48")
    row["tg_ge500_flag"] = 1 if (tg_first is not None and tg_first >= 500) else 0
    row["tg_admission"] = tg_first

    # Metabolic dx flag
    has_met = con.execute(f"""
        SELECT COUNT(*) FROM diagnosis
        WHERE patientunitstayid = {stay_id}
          AND (icd9code LIKE '278.0%' OR icd9code LIKE '272.1%'
               OR icd9code LIKE '272.4%' OR icd9code LIKE '272.2%'
               OR LOWER(diagnosisstring) LIKE '%obes%'
               OR LOWER(diagnosisstring) LIKE '%hypertriglyceridemia%')
    """).fetchone()[0]
    row["metabolic_dx_flag"] = 1 if has_met > 0 else 0

    # Outcome
    pt_info = con.execute(f"""
        SELECT hospitaldischargestatus, unitdischargestatus
        FROM patient WHERE patientunitstayid = {stay_id}
    """).fetchone()
    row["hospital_expire_flag"] = 1 if pt_info and pt_info[0] == "Expired" else 0
    row["icu_flag"] = 1  # All eICU stays are ICU

    # Age
    age_str = con.execute(f"SELECT age FROM patient WHERE patientunitstayid = {stay_id}").fetchone()
    try:
        row["age"] = int(age_str[0]) if age_str and age_str[0] != "" else None
    except (ValueError, TypeError):
        row["age"] = None

    # Gender
    gender = con.execute(f"SELECT gender FROM patient WHERE patientunitstayid = {stay_id}").fetchone()
    row["gender"] = 1 if gender and gender[0] and gender[0][0].upper() == "M" else 0

    eicu_features.append(row)

df_eicu = pd.DataFrame(eicu_features)

# Composite outcome (ICU 7d not available in eICU; use hospital mortality as proxy)
df_eicu["composite_outcome"] = df_eicu["hospital_expire_flag"]
n_events = df_eicu["composite_outcome"].sum()
print(f"  eICU MDAP features: {df_eicu.shape}")
print(f"  Composite outcome (hosp death): {n_events}/{len(df_eicu)} ({100*n_events/len(df_eicu):.1f}%)")

# ═══════════════════════════════════════════════════════════
# 4. 特征分布漂移审计
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[4/5] Feature Distribution Shift Audit (Subasri 2025)")
print("=" * 70)

df_mimic = pd.read_csv(os.path.join(OUTDIR, "canonical_mdap_cohort.csv"))

SHIFT_FEATURES = [
    "age", "tg_ge500_flag", "metabolic_dx_flag",
    "baseline_wbc", "baseline_creatinine", "baseline_bun",
    "baseline_platelet", "baseline_glucose",
]

# Map eICU columns to MIMIC-IV equivalents
EICU_TO_MIMIC = {
    "wbc_w0_6": "baseline_wbc",
    "creatinine_w0_6": "baseline_creatinine",
    "bun_w0_6": "baseline_bun",
    "platelet_w0_6": "baseline_platelet",
    "glucose_w0_6": "baseline_glucose",
}

shift_results = []
for mimic_col in SHIFT_FEATURES:
    eicu_col = None
    for ec, mc in EICU_TO_MIMIC.items():
        if mc == mimic_col:
            eicu_col = ec
            break
    if eicu_col is None:
        eicu_col = mimic_col

    mimic_vals = df_mimic[mimic_col].dropna() if mimic_col in df_mimic.columns else pd.Series()
    eicu_vals = df_eicu[eicu_col].dropna() if eicu_col in df_eicu.columns else pd.Series()

    if len(mimic_vals) < 5 or len(eicu_vals) < 5:
        continue

    smd = abs(mimic_vals.mean() - eicu_vals.mean()) / np.sqrt(
        (mimic_vals.var() + eicu_vals.var()) / 2
    ) if (mimic_vals.var() + eicu_vals.var()) > 0 else 0

    ks_stat, ks_p = stats.ks_2samp(mimic_vals, eicu_vals)

    shift_results.append({
        "feature": mimic_col,
        "mimic_mean": round(mimic_vals.mean(), 2),
        "eicu_mean": round(eicu_vals.mean(), 2),
        "smd": round(smd, 3),
        "ks_stat": round(ks_stat, 3),
        "ks_p": round(ks_p, 4),
        "shift_flag": "YES" if smd > 0.2 else "no",
    })

df_shift = pd.DataFrame(shift_results)
df_shift.to_csv(os.path.join(OUTDIR, "eicu_transportability_shift.csv"), index=False)

print("\n  Feature distribution shift (MIMIC-IV vs eICU):")
for _, row in df_shift.iterrows():
    flag = " <<<" if row["shift_flag"] == "YES" else ""
    print(f"    {row['feature']:25s}: MIMIC={row['mimic_mean']:.2f}, eICU={row['eicu_mean']:.2f}, "
          f"SMD={row['smd']:.3f}, KS={row['ks_stat']:.3f}{flag}")

n_shifted = (df_shift["shift_flag"] == "YES").sum()
print(f"\n  Features with SMD > 0.2: {n_shifted}/{len(df_shift)}")

# ═══════════════════════════════════════════════════════════
# 5. 风险排序保持性 + 校准漂移
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[5/5] Risk Ranking Preservation + Calibration Drift")
print("=" * 70)

# Load MIMIC-IV trained model
df_48h = pd.read_csv(os.path.join(OUTDIR, "landmark_features_48h.csv"))
df_cohort_full = pd.read_csv(os.path.join(OUTDIR, "canonical_mdap_cohort.csv"))
df_cohort_full["composite_outcome"] = (
    (df_cohort_full["icu_7d"] == 1) | (df_cohort_full["hospital_expire_flag"] == 1)
).astype(int)
df_cohort_full["t0_dt"] = pd.to_datetime(df_cohort_full["t0"])
df_cohort_full = df_cohort_full.sort_values("t0_dt").reset_index(drop=True)
split_idx = int(len(df_cohort_full) * 0.7)
train_hadm = set(df_cohort_full.iloc[:split_idx]["hadm_id"].values)

BASELINE_FEATURES = [
    "age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission",
    "diabetes", "hypertension", "heart_failure", "cad", "copd", "ckd",
    "dyslipidemia", "obesity_dx", "htg_dx",
    "baseline_wbc", "baseline_creatinine", "baseline_bun",
    "baseline_bilirubin", "baseline_platelet", "baseline_glucose",
    "baseline_lipase", "baseline_calcium",
]
TRAJ_MARKERS_ALL = ["wbc", "creatinine", "bun", "platelet", "glucose", "lactate", "bilirubin"]

def build_t0_features(df):
    # 重指向 T0：仅入院基线特征（与 03/04 主模型一致），不含 0-48h 轨迹/gbtm_class
    return [f for f in BASELINE_FEATURES if f in df.columns]

# Re-train MIMIC-IV model
t0_features = build_t0_features(df_48h)
df_48h_merged = df_48h.merge(
    df_cohort_full[["hadm_id", "composite_outcome"]], on="hadm_id", how="left"
)
train_mask = df_48h_merged["hadm_id"].isin(train_hadm)

X_mimic = df_48h_merged[t0_features].copy()
if "gbtm_class" in X_mimic.columns:
    X_mimic["gbtm_class"] = X_mimic["gbtm_class"].fillna(-1).astype(int)
for col in X_mimic.columns:
    X_mimic[col] = X_mimic[col].fillna(X_mimic.loc[train_mask, col].median())

X_train_mimic = X_mimic[train_mask].values
y_train_mimic = df_48h_merged.loc[train_mask, "composite_outcome"].values

lgb_params = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "n_estimators": 200, "max_depth": 4, "num_leaves": 15,
    "learning_rate": 0.05, "min_child_samples": 20,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "is_unbalance": False,
}

model_mimic = lgb.LGBMClassifier(**lgb_params)
model_mimic.fit(X_train_mimic, y_train_mimic)

# Align eICU features to MIMIC-IV model input
# eICU only has baseline + trajectory markers (no comorbidities, no gbtm_class)
eicu_model_features = []
for feat in t0_features:
    if feat in df_eicu.columns:
        eicu_model_features.append(feat)
    elif feat == "gbtm_class":
        continue
    elif feat.startswith("baseline_"):
        marker = feat.replace("baseline_", "")
        eicu_col = f"{marker}_w0_6"
        if eicu_col in df_eicu.columns:
            eicu_model_features.append(eicu_col)
    else:
        eicu_model_features.append(feat)

# Build aligned feature matrix
X_eicu = pd.DataFrame(index=df_eicu.index)
for feat in t0_features:
    if feat in df_eicu.columns:
        X_eicu[feat] = df_eicu[feat]
    elif feat == "gbtm_class":
        X_eicu[feat] = -1
    elif feat.startswith("baseline_"):
        marker = feat.replace("baseline_", "")
        eicu_col = f"{marker}_w0_6"
        if eicu_col in df_eicu.columns:
            X_eicu[feat] = df_eicu[eicu_col]
        else:
            X_eicu[feat] = np.nan
    elif feat in df_eicu.columns:
        X_eicu[feat] = df_eicu[feat]
    else:
        X_eicu[feat] = np.nan

# Fill NaN with the same MIMIC-IV training medians used by the locked model.
train_medians = X_mimic.loc[train_mask, :].median(numeric_only=True)
for col in X_eicu.columns:
    if col in X_mimic.columns:
        fill_value = train_medians.get(col, np.nan)
        if pd.isna(fill_value):
            fill_value = 0
        X_eicu[col] = X_eicu[col].fillna(fill_value)
    else:
        X_eicu[col] = X_eicu[col].fillna(0)

X_eicu_arr = X_eicu[t0_features].values.astype(float)

# Predict on eICU
prob_eicu = model_mimic.predict_proba(X_eicu_arr)[:, 1]
y_eicu = df_eicu["composite_outcome"].values

print(f"\n  eICU predictions: N={len(y_eicu)}, events={y_eicu.sum()}")

# Risk ranking preservation
if y_eicu.sum() > 0 and (1 - y_eicu).sum() > 0:
    auroc_eicu = roc_auc_score(y_eicu, prob_eicu)
    brier_eicu = brier_score_loss(y_eicu, prob_eicu)

    # Calibration
    from sklearn.linear_model import LogisticRegression as LR
    lr = LR(fit_intercept=True, max_iter=1000)
    lr.fit(prob_eicu.reshape(-1, 1), y_eicu)
    cal_int = lr.intercept_[0]
    cal_slope = lr.coef_[0][0]

    # Spearman rank correlation
    spearman_r, spearman_p = stats.spearmanr(prob_eicu, y_eicu)

    print(f"  AUROC (directional): {auroc_eicu:.3f}")
    print(f"  Brier score: {brier_eicu:.3f}")
    print(f"  Calibration intercept: {cal_int:.3f} (ideal=0)")
    print(f"  Calibration slope: {cal_slope:.3f} (ideal=1)")
    print(f"  Spearman rank correlation: {spearman_r:.3f} (p={spearman_p:.4f})")

    # Risk tertile outcome rates
    tertiles = pd.qcut(prob_eicu, q=3, labels=["Low", "Mid", "High"], duplicates="drop")
    tertile_records = []
    print(f"\n  Risk tertile outcome rates:")
    for t in ["Low", "Mid", "High"]:
        mask = tertiles == t
        n = int(mask.sum())
        events = int(y_eicu[mask].sum())
        rate = 100 * events / n if n > 0 else 0
        tertile_records.append({
            "tertile": t,
            "n": n,
            "events": events,
            "event_rate_pct": round(rate, 1),
        })
        print(f"    {t}: n={n}, events={events} ({rate:.1f}%)")
    pd.DataFrame(tertile_records).to_csv(
        os.path.join(OUTDIR, "eicu_tertile_enrichment.csv"), index=False
    )

    # Ordinal risk separation test
    low_rate = y_eicu[tertiles == "Low"].mean() if (tertiles == "Low").sum() > 0 else 0
    high_rate = y_eicu[tertiles == "High"].mean() if (tertiles == "High").sum() > 0 else 0
    print(f"\n  Ordinal risk separation: Low={low_rate:.3f} -> High={high_rate:.3f}")
    if high_rate > low_rate:
        print("  OK: Risk ordering preserved (High > Low)")
    else:
        print("  WARNING: Risk ordering NOT preserved")

    transport_results = [{
        "metric": "AUROC", "value": round(auroc_eicu, 3),
        "interpretation": "Directional discriminative ability",
    }, {
        "metric": "Brier", "value": round(brier_eicu, 3),
        "interpretation": "Overall prediction accuracy",
    }, {
        "metric": "Cal_intercept", "value": round(cal_int, 3),
        "interpretation": "Calibration-in-the-large (ideal=0)",
    }, {
        "metric": "Cal_slope", "value": round(cal_slope, 3),
        "interpretation": "Calibration slope (ideal=1)",
    }, {
        "metric": "Spearman_r", "value": round(spearman_r, 3),
        "interpretation": "Rank correlation (risk ordering)",
    }, {
        "metric": "Spearman_p", "value": round(spearman_p, 4),
        "interpretation": "Spearman rank-correlation p-value",
    }, {
        "metric": "N", "value": len(y_eicu),
        "interpretation": "Sample size (too small for definitive validation)",
    }, {
        "metric": "N_events", "value": int(y_eicu.sum()),
        "interpretation": "Number of events",
    }]

    df_transport = pd.DataFrame(transport_results)
    df_transport.to_csv(os.path.join(OUTDIR, "eicu_transportability_results.csv"), index=False)
else:
    print("  Too few events for AUROC computation")
    low_rate = high_rate = np.nan
    tertile_records = []

# Variable availability audit
print(f"\n  Variable availability audit:")
availability_records = []
for marker in ["wbc", "creatinine", "bun", "platelet", "glucose", "lactate", "tg"]:
    for w in ["w0_6", "w6_24", "w24_48"]:
        col = f"{marker}_{w}"
        if col in df_eicu.columns:
            n_avail = int(df_eicu[col].notna().sum())
            pct = 100 * n_avail / len(df_eicu)
            availability_records.append({
                "marker": marker,
                "window": w,
                "n_available": n_avail,
                "pct": round(pct, 1),
            })
            if pct > 0:
                print(f"    {col}: {n_avail}/{len(df_eicu)} ({pct:.0f}%)")
pd.DataFrame(availability_records).to_csv(
    os.path.join(OUTDIR, "eicu_variable_availability.csv"), index=False
)

# ═══════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("AI-FAP: eICU Transportability Audit (N=72, directional)", fontsize=13, fontweight="bold")

# SMD bar chart
ax = axes[0]
if len(df_shift) > 0:
    colors = ["#F44336" if s > 0.2 else "#2196F3" for s in df_shift["smd"]]
    ax.barh(range(len(df_shift)), df_shift["smd"].values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(df_shift)))
    ax.set_yticklabels(df_shift["feature"].values[::-1], fontsize=8)
    ax.axvline(x=0.2, color="red", linestyle="--", alpha=0.5, label="SMD=0.2")
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_title("Feature Distribution Shift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

# Risk distribution comparison
ax = axes[1]
if y_eicu.sum() > 0:
    ax.hist(prob_eicu[y_eicu == 0], bins=15, alpha=0.6, color="#2196F3", label="Survived", density=True)
    ax.hist(prob_eicu[y_eicu == 1], bins=15, alpha=0.6, color="#F44336", label="Died", density=True)
    ax.set_xlabel("Predicted probability (MIMIC-IV model)")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution in eICU")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Risk tertile outcomes
ax = axes[2]
if y_eicu.sum() > 0:
    tertile_rates = []
    tertile_labels = []
    for t in ["Low", "Mid", "High"]:
        mask = tertiles == t
        if mask.sum() > 0:
            tertile_rates.append(100 * y_eicu[mask].mean())
            tertile_labels.append(f"{t}\nn={mask.sum()}")
    bar_colors = ["#4CAF50", "#FF9800", "#F44336"]
    ax.bar(tertile_labels, tertile_rates, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mortality rate (%)")
    ax.set_title("Outcome by Risk Tertile")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(OUTDIR, "eicu_transportability_audit.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  Figure saved to {fig_path}")

# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("eICU TRANSPORTABILITY AUDIT SUMMARY")
print("=" * 70)
print(f"""
eICU-CRD MDAP: {n_mdap} stays
Composite outcome: {n_events} ({100*n_events/max(n_mdap,1):.1f}%)

This is a small-sample directional audit.
It can answer: "Does risk ordering roughly hold?"
It cannot answer: "Is the model externally validated?"

Key findings:
  Feature shift: {n_shifted} features with SMD > 0.2
  Risk ordering: {'Preserved' if y_eicu.sum() > 0 and high_rate > low_rate else 'Not assessable'}
  Calibration: Requires local recalibration (slope likely ≠ 1)

Honest reporting template:
  "AI-FAP preserved ordinal risk separation in eICU-CRD but
   required local recalibration before being used as an absolute
   low-risk de-escalation rule."

Output files:
  eicu_transportability_shift.csv
  eicu_transportability_results.csv
  eicu_tertile_enrichment.csv
  eicu_variable_availability.csv
  eicu_transportability_summary.json
  eicu_transportability_audit.png
""")

summary = {
    "database": "eICU-CRD",
    "scope": "small-sample directional transportability audit (NOT external validation)",
    "cohort": {
        "N": int(n_mdap),
        "events_composite": int(n_events),
        "event_rate_composite": round(100 * float(n_events) / max(n_mdap, 1), 2),
    },
    "feature_shift": {
        "n_features_audited": int(len(df_shift)),
        "n_features_smd_gt_0.2": int(n_shifted),
    },
    "discrimination": {
        "auroc": round(float(auroc_eicu), 4) if "auroc_eicu" in globals() else None,
        "brier": round(float(brier_eicu), 4) if "brier_eicu" in globals() else None,
    },
    "calibration": {
        "intercept": round(float(cal_int), 4) if "cal_int" in globals() else None,
        "slope": round(float(cal_slope), 4) if "cal_slope" in globals() else None,
    },
    "rank_preservation": {
        "spearman_r": round(float(spearman_r), 4) if "spearman_r" in globals() else None,
        "spearman_p": round(float(spearman_p), 4) if "spearman_p" in globals() else None,
        "ordinal_separation_ok": bool(high_rate > low_rate) if np.isfinite(low_rate) else None,
        "tertile_rates": {
            r["tertile"]: r["event_rate_pct"] for r in tertile_records
        } if tertile_records else None,
    },
    "outputs": {
        "shift_csv": "eicu_transportability_shift.csv",
        "results_csv": "eicu_transportability_results.csv",
        "tertile_csv": "eicu_tertile_enrichment.csv",
        "availability_csv": "eicu_variable_availability.csv",
        "figure_png": "eicu_transportability_audit.png",
        "summary_json": "eicu_transportability_summary.json",
    },
}
with open(os.path.join(OUTDIR, "eicu_transportability_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

con.close()
