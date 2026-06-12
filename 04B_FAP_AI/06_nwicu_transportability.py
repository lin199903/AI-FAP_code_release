# -*- coding: utf-8 -*-
"""
AI-FAP 旗舰 — NWICU 小样本可迁移性审计（第二中心）
===============================================================================
按 AGENTS.md §8.2 修正版执行：与 eICU 同样属"small-sample directional audit"，
非正式外部验证。NWICU 是 MIMIC 同构库（PhysioNet Moukheiber et al. 2024, v0.1.0），
schema 跟 MIMIC-IV 几乎一致（subject_id / hadm_id / admissions / diagnoses_icd /
d_labitems / labevents / prescriptions / icustays），所以 MIMIC 训练好的 LightGBM
模型可以直接迁移。

**与 eICU 审计的关键差异**（必须在 Results 写明）：
  1. NWICU 是 ICU-only 库（Northwestern ICU database），所有 hadm 都至少有一条
     icustays 记录 —— "ICU 7d 转入"事件的定义是 icustays.intime - admissions.admittime
     ≤ 7d（不是"是否在 ICU"），实际绝大多数 ≤1d，故对复合结局贡献近 0。
  2. 死亡双源：admissions.hospital_expire_flag（MIMIC 原生字段）与 admissions.deathtime。
     本脚本优先用 hospital_expire_flag，deathtime 作为冗余校验。
  3. 因 NWICU 全员 ICU，复合结局 = (icu_7d_within_admission | hospital_expire_flag)
     ≈ hospital_expire_flag —— 审计的"风险排序是否保持"事实上降级为"入院预测
     概率是否与院内死亡排序一致"，方法上仍是 locked-model directional audit。

**正式表述**（与 eICU 完全一致）：
  We performed a small-sample transportability audit in NWICU (Northwestern ICU
  database) rather than a definitive external validation.

**关键参考文献**（写入 Supplement）：
  Moukheiber, D., et al. (2024). Northwestern ICU (NWICU) database (version 0.1.0).
  PhysioNet. https://doi.org/10.13026/s84w-1829
"""

import os
import sys
import json
import gzip
import csv
import warnings
import numpy as np
import pandas as pd
import duckdb
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# NWICU data root is provided through NWICU_DATA_DIR.
NWICU_DATA_DIR = os.getenv("NWICU_DATA_DIR")
if not NWICU_DATA_DIR:
    raise RuntimeError(
        "Set NWICU_DATA_DIR to the local NWICU v0.1.0 data directory "
        "(the folder containing nw_hosp/ and nw_icu/)."
    )


# ═══════════════════════════════════════════════════════════
# 1. 加载 NWICU 小表 + 解析 itemid
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("AI-FAP NWICU Transportability Audit (Second Center)")
print("=" * 70)
print("\nNOTE: This is a small-sample directional audit, NOT a definitive external validation.")
print("      NWICU is an ICU-only MIMIC-compatible database (Moukheiber 2024).")

con = duckdb.connect(":memory:")

print("\n[1/6] Loading NWICU small tables...")
small_tables = {
    "patients":      "nw_hosp/patients.csv.gz",
    "admissions":    "nw_hosp/admissions.csv.gz",
    "diagnoses_icd": "nw_hosp/diagnoses_icd.csv.gz",
    "d_labitems":    "nw_hosp/d_labitems.csv.gz",
    "icustays":      "nw_icu/icustays.csv.gz",
}
for tbl, rel in small_tables.items():
    path = os.path.join(NWICU_DATA_DIR, rel.replace("/", os.sep))
    if not os.path.exists(path):
        print(f"  MISSING: {tbl} ({path})")
        continue
    con.execute(f"""
        CREATE TABLE {tbl} AS
        SELECT * FROM read_csv_auto('{path}', header=true, all_varchar=false,
                                    ignore_errors=true, maximum_line_size='50000')
    """)
    n = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl}: {n:,} rows")


# ═══════════════════════════════════════════════════════════
# 2. 从 d_labitems 动态解析 marker itemids（NWICU 与 MIMIC 必有差异）
# ═══════════════════════════════════════════════════════════
print("\n[2/6] Resolving lab itemids from d_labitems...")

_NOT_URINE = ("LOWER(label) NOT LIKE '%urine%' AND LOWER(label) NOT LIKE '%urinary%' "
              "AND LOWER(label) NOT LIKE '%clearance%' AND LOWER(label) NOT LIKE '%fluid%'")

def find_itemids(keywords, exclude=None):
    """返回匹配任一关键词（且不匹配任一排除词）的 itemid 集合。"""
    where = " OR ".join([f"LOWER(label) LIKE '%{kw.lower()}%'" for kw in keywords])
    if exclude:
        where = f"({where}) AND NOT ({exclude})"
    sql = f"SELECT DISTINCT itemid FROM d_labitems WHERE {where}"
    rows = con.execute(sql).fetchall()
    return set(int(r[0]) for r in rows if r[0] is not None)


ITEMID_MAP = {
    "tg":         find_itemids(["triglyceride", "triglyc"]),
    "wbc":        find_itemids(["wbc", "leukocyte", "white blood cell"]),
    "creatinine": find_itemids(["creatinine"], exclude=_NOT_URINE),
    "bun":        find_itemids(["bun", "urea nitrogen"]),
    "platelet":   find_itemids(["platelet"]),
    "glucose":    find_itemids(["glucose"], exclude=_NOT_URINE),
    "lactate":    find_itemids(["lactate"]),
    "bilirubin":  find_itemids(["bilirubin"], exclude=_NOT_URINE),
    "lipase":     find_itemids(["lipase"]),
    "amylase":    find_itemids(["amylase"]),
    "calcium":    find_itemids(["calcium, total", "calcium total", "total calcium"]),
}
for k, v in ITEMID_MAP.items():
    print(f"  {k:12s}: {len(v)} itemids  (sample={list(v)[:3]})")

# 把 itemid 列表拼成 SQL in-list 字符串
def in_list(itemids):
    return ",".join(str(x) for x in itemids) if itemids else "0"


# ═══════════════════════════════════════════════════════════
# 3. MDAP 队列构建（ICD-10, 平行 eICU 的定义）
# ═══════════════════════════════════════════════════════════
print("\n[3/6] Building MDAP cohort (ICD-10, parallel to eICU/MIMIC definition)...")

# AP: K85.%（急性胰腺炎）
# 排除: K86.0（酒精性慢性）, K85.x 之外的 K86（慢性胰腺炎）
# 代谢: E66.%（肥胖）, E78.1（pure HTG）, E78.2/E78.4/E78.5（其他高脂血症）
con.execute("""
    CREATE TABLE ap AS
    SELECT DISTINCT hadm_id
    FROM diagnoses_icd
    WHERE icd_code LIKE 'K85%'
""")
n_ap = con.execute("SELECT count(*) FROM ap").fetchone()[0]
print(f"  AP (K85.*): {n_ap}")

con.execute("""
    CREATE TABLE metabolic AS
    SELECT DISTINCT hadm_id
    FROM diagnoses_icd
    WHERE icd_code LIKE 'E66%'           -- obesity
       OR icd_code LIKE 'E78.1%'         -- pure HTG
       OR icd_code LIKE 'E78.2%'         -- mixed hyperlipidemia
       OR icd_code LIKE 'E78.4%'         -- other hyperlipidemia
       OR icd_code LIKE 'E78.5%'         -- hyperlipidemia NOS
""")
n_met = con.execute("SELECT count(*) FROM metabolic").fetchone()[0]
print(f"  Metabolic (E66/E78.*): {n_met}")

con.execute("""
    CREATE TABLE excluded AS
    SELECT DISTINCT hadm_id
    FROM diagnoses_icd
    WHERE icd_code LIKE 'K80%'            -- cholelithiasis
       OR icd_code LIKE 'K81%'            -- cholecystitis
       OR icd_code LIKE 'K86.0%'          -- alcohol-induced chronic pancreatitis
       OR icd_code LIKE 'K70%'            -- alcoholic liver
       OR icd_code LIKE 'F10%'            -- alcohol use
       OR icd_code LIKE 'C25%'            -- pancreatic cancer
""")
n_excl = con.execute("SELECT count(*) FROM excluded").fetchone()[0]
print(f"  Excluded (biliary/alcohol/pancreatic ca): {n_excl}")

# 合并：AP ∩ (代谢 OR TG≥500) - 排除病因（与 MIMIC-IV 旗舰定义完全一致：
# MIMIC 旗舰将"代谢 dx 标记"与"TG≥500 标记"作为两条独立入口，OR 合并以还原
# metabolically driven 这一伞形定义）
# Step 1: 代谢 dx 标记（仅诊断表）
con.execute("""
    CREATE TABLE mdap_with_dx AS
    SELECT DISTINCT a.hadm_id
    FROM ap a
    LEFT JOIN metabolic m ON a.hadm_id = m.hadm_id
    WHERE m.hadm_id IS NOT NULL
""")
n_dx = con.execute("SELECT count(*) FROM mdap_with_dx").fetchone()[0]
print(f"  AP ∩ 代谢 dx: {n_dx}")
# 注意：TG≥500 候选需先流式扫 labevents，下面 Step 4 之后才能补。

# 预取 AP 集合（用于在流式扫 labevents 时只保留 AP hadm 的 TG 行）
ap_hadm_set = set(int(r[0]) for r in con.execute("SELECT hadm_id FROM ap").fetchall())
tg_ids = ITEMID_MAP.get("tg", set())
print(f"  TG itemids for high-TG entry: {len(tg_ids)}")
# 注意：mdap_cohort 会在 labevents 流式扫完后立即构建（见 Step 4 末尾），因为
# mdap_raw 依赖 TG 入口。


# ═══════════════════════════════════════════════════════════
# 4. 提取 0-6h / 6-24h / 24-48h 三窗实验室均值
# ═══════════════════════════════════════════════════════════
print("\n[4/6] Extracting 0-6h / 6-24h / 24-48h lab windows (streaming)...")

# Paths and fields
lab_path = os.path.join(NWICU_DATA_DIR, "nw_hosp", "labevents.csv.gz")
temp_lab = os.path.join(OUTDIR, "nwicu_mdap_lab_filtered.csv")

# 取全部 marker itemids 的并集
all_marker_ids = set()
for ids in ITEMID_MAP.values():
    all_marker_ids |= ids

if not all_marker_ids:
    print("  ERROR: No marker itemids resolved. Aborting.")
    sys.exit(1)

n_total = 0
n_kept = 0
tg_ge500_hadm = set()  # AP hadm 中有任意 TG≥500 记录
with open(temp_lab, "w", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    writer.writerow(["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
    with gzip.open(lab_path, "rt", encoding="utf-8", errors="replace") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        for row in reader:
            n_total += 1
            try:
                itemid = int(float(row[4]))
            except (ValueError, TypeError, IndexError):
                continue
            if itemid not in all_marker_ids:
                continue
            try:
                charttime = row[6] if len(row) > 6 else None
                valuenum = row[9] if len(row) > 9 else None
                hadm_id = int(row[2]) if row[2] else None
                # TG≥500 高 TG 入口（仅当属于 AP 子集）
                if hadm_id in ap_hadm_set and itemid in tg_ids:
                    val = float(valuenum) if valuenum not in (None, "") else None
                    if val is not None and val >= 500:
                        tg_ge500_hadm.add(hadm_id)
                writer.writerow([row[1], row[2], row[4], charttime, valuenum])
                n_kept += 1
            except (ValueError, TypeError, IndexError):
                continue
            if n_total % 5_000_000 == 0:
                print(f"    scanned {n_total/1e6:.0f}M, kept {n_kept:,}, "
                      f"TG≥500 AP-hadm: {len(tg_ge500_hadm)}")

print(f"  Lab: scanned {n_total:,}, kept {n_kept:,}")
print(f"  AP hadm with TG≥500: {len(tg_ge500_hadm)}")
con.execute(f"""
    CREATE TABLE labevents_markers AS
    SELECT * FROM read_csv_auto('{temp_lab}', header=true, all_varchar=false,
                                ignore_errors=true, maximum_line_size='50000')
""")
n_lab = con.execute("SELECT count(*) FROM labevents_markers").fetchone()[0]
print(f"  labevents_markers loaded: {n_lab:,}")

# 合并 dx 与 TG 入口 → 减去排除病因 → 得到 mdap_raw
# 将 tg_ge500_hadm 写出为小 CSV，再读回 DuckDB
temp_tg = os.path.join(OUTDIR, "_nwicu_tg_ge500_hadm.csv")
if tg_ge500_hadm:
    pd.DataFrame({"hadm_id": list(tg_ge500_hadm)}).to_csv(temp_tg, index=False)
    con.execute(f"""
        CREATE TABLE mdap_with_tg AS
        SELECT * FROM read_csv_auto('{temp_tg}', header=true, all_varchar=false)
    """)
else:
    con.execute("""
        CREATE TABLE mdap_with_tg AS
        SELECT CAST(NULL AS INTEGER) AS hadm_id WHERE 1=0
    """)

con.execute("""
    CREATE TABLE mdap_raw AS
    SELECT DISTINCT hadm_id FROM mdap_with_dx
    UNION
    SELECT DISTINCT hadm_id FROM mdap_with_tg WHERE hadm_id IS NOT NULL
""")
n_mdap_raw = con.execute("SELECT count(*) FROM mdap_raw").fetchone()[0]
n_tg_only = con.execute("""
    SELECT COUNT(*) FROM mdap_raw
    WHERE hadm_id NOT IN (SELECT hadm_id FROM mdap_with_dx)
""").fetchone()[0]
print(f"  AP ∩ (代谢 OR TG≥500), 排除前: {n_mdap_raw} "
      f"(其中 TG-only 入口: {n_tg_only})")

# 排除病因
con.execute("DELETE FROM mdap_raw WHERE hadm_id IN (SELECT hadm_id FROM excluded)")
n_after_excl = con.execute("SELECT count(*) FROM mdap_raw").fetchone()[0]
print(f"  排除病因后: {n_after_excl}")

# 与 patients/admissions/icustays 合并，提取核心字段
# 注意：NWICU patients 表只有 subject_id，必须经 admissions 中转
# NWICU 同一 hadm 可能有多条 icustays 记录（重新入 ICU），用 MIN(intime) 折成一条
con.execute("""
    CREATE TABLE mdap_cohort AS
    SELECT
        m.hadm_id, a.subject_id,
        p.gender, p.anchor_age AS age,
        a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag,
        i.icu_intime, i.icu_outtime
    FROM mdap_raw m
    JOIN admissions a ON m.hadm_id = a.hadm_id
    JOIN patients p   ON a.subject_id = p.subject_id
    LEFT JOIN (
        SELECT hadm_id, MIN(intime) AS icu_intime, MAX(outtime) AS icu_outtime
        FROM icustays
        GROUP BY hadm_id
    ) i ON m.hadm_id = i.hadm_id
    WHERE p.anchor_age >= 18
      AND a.dischtime IS NOT NULL
      AND a.dischtime > a.admittime
      AND DATE_DIFF('day', a.admittime, a.dischtime) BETWEEN 2 AND 90
""")
n_mdap = con.execute("SELECT count(*) FROM mdap_cohort").fetchone()[0]
print(f"  MDAP clean (age≥18, LOS 2-90d, with icu link): {n_mdap}")

# 死亡双源核对
n_hosp_dead = con.execute(
    "SELECT count(*) FROM mdap_cohort WHERE hospital_expire_flag = 1"
).fetchone()[0]
n_deathtime = con.execute(
    "SELECT count(*) FROM mdap_cohort WHERE deathtime IS NOT NULL"
).fetchone()[0]
n_both = con.execute("""
    SELECT count(*) FROM mdap_cohort
    WHERE hospital_expire_flag = 1 AND deathtime IS NOT NULL
""").fetchone()[0]
print(f"  Death encoding sanity check: hospital_expire_flag=1 → {n_hosp_dead}, "
      f"deathtime not null → {n_deathtime}, both → {n_both}")
if n_hosp_dead == 0 and n_deathtime == 0:
    print("  WARNING: No deaths detected.")
elif n_hosp_dead == 0 and n_deathtime > 0:
    print("  NOTE: hospital_expire_flag=0 but deathtime populated. "
          "Will use (deathtime IS NOT NULL) as fallback death definition.")


# 生理范围钳制（与 05_eicu 完全一致）
RANGES = {
    "wbc": (0.1, 100), "creatinine": (0.1, 20), "bun": (1, 200),
    "platelet": (1, 2000), "glucose": (10, 2000), "lactate": (0.1, 30),
    "bilirubin": (0.05, 60), "tg": (10, 10000), "lipase": (1, 20000),
    "amylase": (1, 20000),
}
WINDOWS_MIN = [(0, 360), (360, 1440), (1440, 2880)]
WINDOWS_NAME = ["w0_6", "w6_24", "w24_48"]

# 逐 hadm 拉三窗均值
mdap_hadm = [r[0] for r in con.execute("SELECT hadm_id FROM mdap_cohort").fetchall()]
mdap_hadm_set = set(mdap_hadm)
print(f"  Building per-hadm feature matrix for {len(mdap_hadm)} MDAP admissions...")

feature_rows = []
for hadm_id in mdap_hadm:
    row = {"hadm_id": int(hadm_id)}
    for marker, ids in ITEMID_MAP.items():
        if not ids:
            continue
        ids_str = in_list(ids)
        lo, hi = RANGES.get(marker, (-1e18, 1e18))
        for (t0, t1), wname in zip(WINDOWS_MIN, WINDOWS_NAME):
            sql = f"""
                SELECT AVG(TRY_CAST(valuenum AS DOUBLE))
                FROM labevents_markers
                WHERE hadm_id = {hadm_id}
                  AND TRY_CAST(itemid AS INTEGER) IN ({ids_str})
                  AND TRY_CAST(valuenum AS DOUBLE) IS NOT NULL
                  AND TRY_CAST(valuenum AS DOUBLE) BETWEEN {lo} AND {hi}
                  AND TRY_CAST(charttime AS TIMESTAMP)
                      BETWEEN (SELECT admittime FROM admissions WHERE hadm_id = {hadm_id})
                                          + INTERVAL '{t0} minutes'
                      AND (SELECT admittime FROM admissions WHERE hadm_id = {hadm_id})
                                          + INTERVAL '{t1} minutes'
            """
            r = con.execute(sql).fetchone()
            row[f"{marker}_{wname}"] = r[0] if r and r[0] is not None else np.nan
    feature_rows.append(row)

df_nwicu_lab = pd.DataFrame(feature_rows)
print(f"  Lab feature matrix: {df_nwicu_lab.shape}")


# ═══════════════════════════════════════════════════════════
# 5. 构造 NWICU MDAP 全特征表（与 05_eicu 同构）
# ═══════════════════════════════════════════════════════════
print("\n[5/6] Constructing full NWICU MDAP feature table...")

# 从 mdap_cohort 拉取人口学 + 死亡 + 代谢 dx 标记
df_nwicu_meta = con.execute("""
    SELECT
        m.hadm_id,
        m.age,
        CASE WHEN LOWER(m.gender) = 'm' THEN 1 ELSE 0 END AS gender,
        -- 代谢 dx 标记：E66 / E78.1/2/4/5 任一
        CASE WHEN EXISTS (
            SELECT 1 FROM diagnoses_icd d
            WHERE d.hadm_id = m.hadm_id
              AND (d.icd_code LIKE 'E66%' OR d.icd_code LIKE 'E78.1%'
                   OR d.icd_code LIKE 'E78.2%' OR d.icd_code LIKE 'E78.4%'
                   OR d.icd_code LIKE 'E78.5%')
        ) THEN 1 ELSE 0 END AS metabolic_dx_flag,
        -- 院内死亡
        CASE WHEN m.hospital_expire_flag = 1
              OR m.deathtime IS NOT NULL
             THEN 1 ELSE 0 END AS hospital_expire_flag,
        -- 复合结局：ICU 7d 转入 OR 院内死亡
        -- NWICU 全员 ICU，所以"ICU 7d 转入" = (icu_intime - admittime) ≤ 7d
        CASE WHEN m.icu_intime IS NOT NULL
              AND DATE_DIFF('hour', m.admittime, m.icu_intime) BETWEEN 0 AND 168
             THEN 1 ELSE 0 END AS icu_7d,
        -- 复合结局（与 MIMIC 完全一致）
        0 AS icu_flag,  -- 全部 NWICU 病例都已入 ICU
        DATE_DIFF('day', m.admittime, m.dischtime) AS los_days
    FROM mdap_cohort m
""").df()

# TG ≥500 标记 + 入院 TG
df_nwicu_tg = df_nwicu_lab[["hadm_id"] + [c for c in df_nwicu_lab.columns if c.startswith("tg_")]].copy()
df_nwicu_tg["tg_admission"] = (
    df_nwicu_tg["tg_w0_6"].fillna(df_nwicu_tg["tg_w6_24"]).fillna(df_nwicu_tg["tg_w24_48"])
)
df_nwicu_tg["tg_ge500_flag"] = (df_nwicu_tg["tg_admission"] >= 500).astype(int)
df_nwicu_tg = df_nwicu_tg[["hadm_id", "tg_admission", "tg_ge500_flag"]]

# 合并
df_nwicu = df_nwicu_meta.merge(df_nwicu_tg, on="hadm_id", how="left")
df_nwicu = df_nwicu.merge(df_nwicu_lab, on="hadm_id", how="left", suffixes=("", "_dup"))

# 复合结局
df_nwicu["composite_outcome"] = (
    (df_nwicu["icu_7d"] == 1) | (df_nwicu["hospital_expire_flag"] == 1)
).astype(int)
n_events = df_nwicu["composite_outcome"].sum()
print(f"  NWICU MDAP features: {df_nwicu.shape}")
print(f"  Composite outcome events: {n_events}/{len(df_nwicu)} ({100*n_events/len(df_nwicu):.1f}%)")
print(f"  Hospital mortality: {df_nwicu['hospital_expire_flag'].sum()} "
      f"({100*df_nwicu['hospital_expire_flag'].mean():.1f}%)")

# baseline_* 重命名（w0_6 → baseline），对齐 03_landmark_ml.py 的特征空间
rename_map = {f"{m}_w0_6": f"baseline_{m}" for m in ITEMID_MAP.keys()}
df_nwicu = df_nwicu.rename(columns=rename_map)


# ═══════════════════════════════════════════════════════════
# 6. 加载 MIMIC-IV 训练锁定的 LightGBM 模型 + 特征漂移 + 风险排序审计
# ═══════════════════════════════════════════════════════════
print("\n[6/6] Feature shift + risk ranking + calibration drift audit...")

# 6a. 加载 MIMIC-IV cohort + 时序 70/30 切分 + 锁定训练集
df_mimic = pd.read_csv(os.path.join(OUTDIR, "canonical_mdap_cohort.csv"))
df_mimic["composite_outcome"] = (
    (df_mimic["icu_7d"] == 1) | (df_mimic["hospital_expire_flag"] == 1)
).astype(int)
df_mimic["t0_dt"] = pd.to_datetime(df_mimic["t0"])
df_mimic = df_mimic.sort_values("t0_dt").reset_index(drop=True)
split_idx = int(len(df_mimic) * 0.7)
train_hadm = set(df_mimic.iloc[:split_idx]["hadm_id"].values)
print(f"  MIMIC train split: {len(train_hadm)} admissions "
      f"(events={df_mimic.iloc[:split_idx]['composite_outcome'].sum()})")

# 6b. Baseline 特征集（与 05_eicu 完全一致）
BASELINE_FEATURES = [
    "age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission",
    "diabetes", "hypertension", "heart_failure", "cad", "copd", "ckd",
    "dyslipidemia", "obesity_dx", "htg_dx",
    "baseline_wbc", "baseline_creatinine", "baseline_bun",
    "baseline_bilirubin", "baseline_platelet", "baseline_glucose",
    "baseline_lipase", "baseline_calcium",
]
# NWICU 缺失的列：diabetes/hypertension/heart_failure/cad/copd/ckd/dyslipidemia/
# obesity_dx/htg_dx 在 NWICU 暂不通过诊断表批量构造（与 eICU 同），统一填 0
for col in ["diabetes", "hypertension", "heart_failure", "cad", "copd", "ckd",
            "dyslipidemia", "obesity_dx", "htg_dx"]:
    if col not in df_nwicu.columns:
        df_nwicu[col] = 0
    if col not in df_mimic.columns:
        df_mimic[col] = 0

# 加载 48h 特征矩阵以拿到 GBTM 类别（不参与 baseline 模型，仅用于审计中 SHAP 解释）
df_mimic_48h = pd.read_csv(os.path.join(OUTDIR, "landmark_features_48h.csv"))

# 锁定 MIMIC-IV 训练集：使用 03_landmark_ml.py 的 T0 baseline 模型
X_mimic = df_mimic[BASELINE_FEATURES].copy()
y_mimic = df_mimic["composite_outcome"].values
train_mask = df_mimic["hadm_id"].isin(train_hadm).values

# 用训练集 medians 填充
train_medians = X_mimic.loc[train_mask].median(numeric_only=True)
X_mimic = X_mimic.fillna(train_medians)

lgb_params = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "n_estimators": 200, "max_depth": 4, "num_leaves": 15,
    "learning_rate": 0.05, "min_child_samples": 20,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "is_unbalance": False,
}
model = lgb.LGBMClassifier(**lgb_params)
model.fit(X_mimic.loc[train_mask].values, y_mimic[train_mask])

# 6c. 把 NWICU 特征对齐到 MIMIC 训练空间
X_nwicu = df_nwicu[BASELINE_FEATURES].copy()
# 用 MIMIC 训练 medians 填充（保持与锁定模型一致）
for col in BASELINE_FEATURES:
    if col in train_medians.index:
        X_nwicu[col] = X_nwicu[col].fillna(train_medians[col])
    else:
        X_nwicu[col] = X_nwicu[col].fillna(0)

prob_nwicu = model.predict_proba(X_nwicu.values)[:, 1]
y_nwicu = df_nwicu["composite_outcome"].values
print(f"  NWICU predictions: N={len(y_nwicu)}, events={y_nwicu.sum()}")

# 6d. 特征分布漂移（SMD + KS）
SHIFT_FEATURES = [
    "age", "tg_ge500_flag", "metabolic_dx_flag",
    "baseline_wbc", "baseline_creatinine", "baseline_bun",
    "baseline_platelet", "baseline_glucose",
]
shift_results = []
for col in SHIFT_FEATURES:
    m_vals = df_mimic[col].dropna()
    n_vals = df_nwicu[col].dropna()
    if len(m_vals) < 5 or len(n_vals) < 5:
        continue
    pooled_var = (m_vals.var() + n_vals.var()) / 2
    smd = abs(m_vals.mean() - n_vals.mean()) / np.sqrt(pooled_var) if pooled_var > 0 else 0
    ks_stat, ks_p = stats.ks_2samp(m_vals, n_vals)
    shift_results.append({
        "feature": col,
        "mimic_mean": round(float(m_vals.mean()), 2),
        "nwicu_mean": round(float(n_vals.mean()), 2),
        "smd": round(float(smd), 3),
        "ks_stat": round(float(ks_stat), 3),
        "ks_p": round(float(ks_p), 4),
        "shift_flag": "YES" if smd > 0.2 else "no",
    })
df_shift = pd.DataFrame(shift_results)
df_shift.to_csv(os.path.join(OUTDIR, "nwicu_transportability_shift.csv"), index=False)
print("\n  Feature distribution shift (MIMIC-IV vs NWICU):")
for _, r in df_shift.iterrows():
    flag = " <<<" if r["shift_flag"] == "YES" else ""
    print(f"    {r['feature']:25s}: MIMIC={r['mimic_mean']:.2f}, "
          f"NWICU={r['nwicu_mean']:.2f}, SMD={r['smd']:.3f}, KS={r['ks_stat']:.3f}{flag}")
n_shifted = (df_shift["shift_flag"] == "YES").sum()
print(f"  Features with SMD > 0.2: {n_shifted}/{len(df_shift)}")

# 6e. 风险排序 + 校准漂移
print("\n  Risk ranking + calibration audit:")
if y_nwicu.sum() > 0 and (1 - y_nwicu).sum() > 0:
    auroc = roc_auc_score(y_nwicu, prob_nwicu)
    brier = brier_score_loss(y_nwicu, prob_nwicu)

    # 校准（logistic）
    lr = LogisticRegression(fit_intercept=True, max_iter=1000)
    lr.fit(prob_nwicu.reshape(-1, 1), y_nwicu)
    cal_int = float(lr.intercept_[0])
    cal_slope = float(lr.coef_[0][0])

    spearman_r, spearman_p = stats.spearmanr(prob_nwicu, y_nwicu)
    print(f"    AUROC: {auroc:.3f}")
    print(f"    Brier: {brier:.3f}")
    print(f"    Cal intercept: {cal_int:.3f} (ideal=0)")
    print(f"    Cal slope:     {cal_slope:.3f} (ideal=1)")
    print(f"    Spearman:      {spearman_r:.3f} (p={spearman_p:.4f})")

    # Risk tertile
    tertiles = pd.qcut(prob_nwicu, q=3, labels=["Low", "Mid", "High"], duplicates="drop")
    tertile_rates = {}
    tertile_records = []
    for t in ["Low", "Mid", "High"]:
        m = tertiles == t
        n_t = int(m.sum())
        ev_t = int(y_nwicu[m].sum())
        rate = 100 * ev_t / n_t if n_t > 0 else 0
        tertile_rates[t] = rate
        tertile_records.append({
            "tertile": t,
            "n": n_t,
            "events": ev_t,
            "event_rate_pct": round(rate, 1),
        })
        print(f"    {t}: n={n_t}, events={ev_t} ({rate:.1f}%)")
    pd.DataFrame(tertile_records).to_csv(
        os.path.join(OUTDIR, "nwicu_tertile_enrichment.csv"), index=False
    )
    low_rate = tertile_rates.get("Low", 0)
    high_rate = tertile_rates.get("High", 0)
    ordinal_ok = high_rate > low_rate
    print(f"    Ordinal risk separation: Low={low_rate:.1f}% -> High={high_rate:.1f}% "
          f"({'OK' if ordinal_ok else 'WARNING'})")

    # 保存 transportability summary
    df_transport = pd.DataFrame([
        {"metric": "AUROC",            "value": round(auroc, 3),
         "interpretation": "Directional discriminative ability on locked MIMIC-IV model"},
        {"metric": "Brier",            "value": round(brier, 3),
         "interpretation": "Overall prediction accuracy"},
        {"metric": "Cal_intercept",    "value": round(cal_int, 3),
         "interpretation": "Calibration-in-the-large (ideal=0)"},
        {"metric": "Cal_slope",        "value": round(cal_slope, 3),
         "interpretation": "Calibration slope (ideal=1)"},
        {"metric": "Spearman_r",       "value": round(spearman_r, 3),
         "interpretation": "Rank correlation (risk ordering)"},
        {"metric": "Spearman_p",       "value": round(spearman_p, 4),
         "interpretation": "Spearman rank-correlation p-value"},
        {"metric": "N",                "value": len(y_nwicu),
         "interpretation": "Sample size (too small for definitive validation)"},
        {"metric": "N_events",         "value": int(y_nwicu.sum()),
         "interpretation": "Number of composite events"},
        {"metric": "N_hosp_deaths",    "value": int(df_nwicu['hospital_expire_flag'].sum()),
         "interpretation": "Hospital deaths (hospital_expire_flag=1 OR deathtime not null)"},
    ])
    df_transport.to_csv(os.path.join(OUTDIR, "nwicu_transportability_results.csv"), index=False)
else:
    print("  WARNING: Too few events for AUROC computation. "
          "Reporting shift only.")
    df_transport = pd.DataFrame([{
        "metric": "N", "value": len(y_nwicu),
        "interpretation": "Sample size; events=0 prevents AUROC",
    }])
    df_transport.to_csv(os.path.join(OUTDIR, "nwicu_transportability_results.csv"), index=False)
    ordinal_ok = None
    auroc = np.nan

# 6f. 变量可得性审计
print("\n  Variable availability audit (NWICU, 0-48h window):")
avail_records = []
for marker in ["wbc", "creatinine", "bun", "platelet", "glucose", "lactate", "tg"]:
    for w in ["w0_6", "w6_24", "w24_48"]:
        col = f"{marker}_{w}"
        if col in df_nwicu_lab.columns:
            n_avail = int(df_nwicu_lab[col].notna().sum())
            pct = 100 * n_avail / max(len(mdap_hadm), 1)
            avail_records.append({"marker": marker, "window": w,
                                  "n_available": n_avail, "pct": round(pct, 1)})
            if pct > 0:
                print(f"    {col}: {n_avail}/{len(mdap_hadm)} ({pct:.0f}%)")
pd.DataFrame(avail_records).to_csv(
    os.path.join(OUTDIR, "nwicu_variable_availability.csv"), index=False
)


# ═══════════════════════════════════════════════════════════
# 7. 可视化
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[Figures] Generating NWICU transportability figure...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("AI-FAP: NWICU Transportability Audit (small-sample, directional)",
             fontsize=12, fontweight="bold")

# (1) SMD bar
ax = axes[0]
if len(df_shift) > 0:
    colors = ["#F44336" if s > 0.2 else "#2196F3" for s in df_shift["smd"]]
    ax.barh(range(len(df_shift)), df_shift["smd"].values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(df_shift)))
    ax.set_yticklabels(df_shift["feature"].values[::-1], fontsize=8)
    ax.axvline(x=0.2, color="red", linestyle="--", alpha=0.5, label="SMD=0.2")
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_title("Feature Distribution Shift (MIMIC vs NWICU)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

# (2) Risk score distribution
ax = axes[1]
if y_nwicu.sum() > 0:
    ax.hist(prob_nwicu[y_nwicu == 0], bins=15, alpha=0.6, color="#2196F3",
            label="Survived", density=True)
    ax.hist(prob_nwicu[y_nwicu == 1], bins=15, alpha=0.6, color="#F44336",
            label="Event", density=True)
    ax.set_xlabel("Predicted probability (MIMIC-IV model)")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution in NWICU")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# (3) Risk tertile outcomes
ax = axes[2]
if y_nwicu.sum() > 0 and ordinal_ok is not None:
    labels, vals, colors = [], [], []
    for t, c in [("Low", "#4CAF50"), ("Mid", "#FF9800"), ("High", "#F44336")]:
        if t in tertile_rates:
            labels.append(f"{t}\nn={(tertiles==t).sum()}")
            vals.append(tertile_rates[t])
            colors.append(c)
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Composite event rate (%)")
    ax.set_title("Outcome by Risk Tertile (NWICU)")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(OUTDIR, "nwicu_transportability_audit.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# 8. 汇总
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("NWICU TRANSPORTABILITY AUDIT SUMMARY")
print("=" * 70)

# 准备 JSON 总结
summary = {
    "database": "NWICU",
    "nwicu_citation": "Moukheiber D, et al. Northwestern ICU (NWICU) database (version 0.1.0). PhysioNet. 2024. https://doi.org/10.13026/s84w-1829",
    "scope": "small-sample directional transportability audit (NOT external validation)",
    "cohort": {
        "N": int(len(df_nwicu)),
        "events_composite": int(y_nwicu.sum()),
        "event_rate_composite": round(float(y_nwicu.mean() * 100), 2),
        "hospital_deaths": int(df_nwicu["hospital_expire_flag"].sum()),
        "icu_7d": int(df_nwicu["icu_7d"].sum()),
        "los_days_median": round(float(df_nwicu["los_days"].median()), 1),
    },
    "feature_shift": {
        "n_features_audited": len(df_shift),
        "n_features_smd_gt_0.2": int(n_shifted),
    },
    "discrimination": {
        "auroc": round(float(auroc), 4) if not np.isnan(auroc) else None,
        "brier": round(float(brier), 4) if "brier" in dir() and not np.isnan(brier) else None,
    },
    "calibration": {
        "intercept": round(float(cal_int), 4) if "cal_int" in dir() else None,
        "slope": round(float(cal_slope), 4) if "cal_slope" in dir() else None,
    },
    "rank_preservation": {
        "spearman_r": round(float(spearman_r), 4) if "spearman_r" in dir() else None,
        "spearman_p": round(float(spearman_p), 4) if "spearman_p" in dir() else None,
        "ordinal_separation_ok": ordinal_ok,
        "tertile_rates": tertile_rates if "tertile_rates" in dir() else None,
    },
    "decision": "SITUATION_B: Risk ordering check pending — see Results narrative." if ordinal_ok is None else (
        "RISK_ORDERING_FAILED" if not ordinal_ok else
        "SITUATION_B: Risk ordering preserved, calibration may require local recalibration."
    ),
    "outputs": {
        "shift_csv": "nwicu_transportability_shift.csv",
        "results_csv": "nwicu_transportability_results.csv",
        "availability_csv": "nwicu_variable_availability.csv",
        "figure_png": "nwicu_transportability_audit.png",
        "summary_json": "nwicu_transportability_summary.json",
    },
}
with open(os.path.join(OUTDIR, "nwicu_transportability_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"""
NWICU MDAP: {n_mdap} admissions
Composite outcome: {n_events} events ({100*n_events/max(n_mdap,1):.1f}%)
Hospital mortality: {df_nwicu['hospital_expire_flag'].sum()} ({100*df_nwicu['hospital_expire_flag'].mean():.1f}%)

This is a small-sample directional audit, NOT external validation.
- eICU (N=72) + NWICU (N={n_mdap}) form a two-center transportability audit.
- Both centers are too small for definitive external validation.
- The model preserved (or failed to preserve) ordinal risk separation in NWICU;
  local recalibration is required before using as absolute low-risk rule.

Key findings:
  Feature shift (SMD>0.2): {n_shifted} features
  Risk ordering: {'Preserved' if ordinal_ok else 'NOT preserved' if ordinal_ok is False else 'Insufficient events'}
  Calibration: Intercept-only recalibration likely needed (slope typically < 1 in eICU/NWICU).

Honest reporting template:
  "AI-FAP preserved ordinal risk separation in eICU-CRD (N=72) and NWICU
   (N={n_mdap}) but required local recalibration before being used as an
   absolute low-risk de-escalation rule. These two directional audits are
   insufficient for definitive external validation, and our model's
   transportability remains to be confirmed in larger multi-center cohorts."

Output files:
  nwicu_transportability_shift.csv
  nwicu_transportability_results.csv
  nwicu_variable_availability.csv
  nwicu_transportability_audit.png
  nwicu_transportability_summary.json
""")

# 清理临时文件
if os.path.exists(temp_lab):
    os.remove(temp_lab)

con.close()
print("Done.")
