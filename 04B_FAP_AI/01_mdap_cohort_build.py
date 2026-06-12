# -*- coding: utf-8 -*-
"""
AI-FAP 旗舰 — MDAP 队列构建 + 基线特征表 (Table 1) + 48h 序列特征工程 + 结局提取
===============================================================================
按 AGENTS.md §8.3（审计后修订版）构建代谢驱动型急性胰腺炎（MDAP）规范分析队列。
输出 canonical MDAP 宽表（每 hadm_id 一行）与 Table 1 汇总 CSV。

数据门控（已通过）：
  G1=226  → TG 清除轨迹版本否决；旗舰主轴改为炎症/器官功能轨迹
  G2=655  → MDAP 表型队列可用（TG≥500 193 + 肥胖/HTG dx 540）
  G4 密集 → WBC/Cr/BUN/血小板/血糖/乳酸/胆红素/脂肪酶 48h 覆盖率高

主轴设计（已锁定）：
  炎症/器官功能轨迹（WBC、乳酸、Cr/BUN、胆红素、血小板、血糖）
  时间窗：0-6h / 6-24h / 24-48h（三窗序列，供 GBTM 输入）
  TG 仅作入院基线表型变量（非轨迹主轴）

连接：psycopg2 → MIMIC-IV PostgreSQL（通过 PG_DSN 或 MDAP_ENV_FILE 配置）
输出：04B_FAP_AI/outputs/  → canonical_mdap_cohort.csv + table1_summary.csv
"""

import os
import csv
import psycopg2
import psycopg2.extras
from _dbconfig import get_dsn

# ── 连接 ─────────────────────────────────────────────────
DSN = get_dsn()
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

conn = psycopg2.connect(DSN)
conn.autocommit = True

# ── ItemID 映射 ──────────────────────────────────────────
LAB_ITEMIDS = {
    "wbc":       (51300, 51301, 51755, 51756),
    "lactate":   (50813, 50843, 50954, 51054, 51795, 51944),
    "creatinine":(50841, 50912, 51021, 51032, 51052, 51067),
    "bun":       (50851, 51006, 51045, 51104, 51804, 51825),
    "bilirubin": (50838, 50885, 51028, 51049, 51783, 51812),
    "platelet":  (51265, 53189),
    "glucose":   (50809, 50842, 50931, 51022, 51034, 51053),
    "lipase":    (50844, 50956, 51036, 51055),
    "amylase":   (50836, 50867, 51020, 51026, 51047, 51072),
    "calcium":   (50893,),
    "tg":        (51000,),         # Blood Triglycerides only
    "alt":       (50861,),
    "ast":       (50878,),
}

def inlist(name):
    """生成 SQL IN 列表，如 '(51300,51301,51755,51756)'"""
    return "(" + ",".join(str(i) for i in LAB_ITEMIDS[name]) + ")"

# ── ICD 编码 ──────────────────────────────────────────────
AP_ICD_WHERE = (
    "(d.icd_version = 9 AND d.icd_code = '5770') "
    "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) = 'K85')"
)

# 排除病因 ICD
EXCLUDE_ICD_WHERE = (
    # 胆源性: ICD9 574.x / ICD10 K80.x
    "(d.icd_version = 9 AND LEFT(d.icd_code, 3) = '574') "
    "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) = 'K80') "
    # 酒精性: ICD9 303.x, 305.0 / ICD10 F10.x
    "OR (d.icd_version = 9 AND (LEFT(d.icd_code, 3) = '303' OR d.icd_code = '3050')) "
    "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) = 'F10') "
    # 外伤: ICD9 860-869, 900-904, 925-929, 940-959 / ICD10 S00-S99, T07-T34
    "OR (d.icd_version = 9 AND "
    "    (d.icd_code BETWEEN '860' AND '86999' "
    "     OR d.icd_code BETWEEN '9000' AND '90499' "
    "     OR d.icd_code BETWEEN '9250' AND '92999' "
    "     OR d.icd_code BETWEEN '940' AND '95999')) "
    "OR (d.icd_version = 10 AND (LEFT(d.icd_code, 1) = 'S' OR LEFT(d.icd_code, 2) IN ('T0','T1'))) "
    # ERCP后: ICD9 997.4 / ICD10 K91.86
    "OR (d.icd_version = 9 AND d.icd_code = '9974') "
    "OR (d.icd_version = 10 AND d.icd_code = 'K9186') "
    # 胰腺癌: ICD9 157.x / ICD10 C25.x
    "OR (d.icd_version = 9 AND LEFT(d.icd_code, 3) = '157') "
    "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) = 'C25') "
    # 妊娠: ICD9 630-679 / ICD10 O00-O99, Z33-Z34
    "OR (d.icd_version = 9 AND d.icd_code BETWEEN '630' AND '67999') "
    "OR (d.icd_version = 10 AND (LEFT(d.icd_code, 1) = 'O' OR d.icd_code IN ('Z33','Z330','Z34','Z340'))) "
)

# 合并症 ICD 编码
COMORB_WHERE = {
    "diabetes": (
        "(d.icd_version = 9 AND LEFT(d.icd_code, 3) BETWEEN '249' AND '250') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) BETWEEN 'E08' AND 'E13')"
    ),
    "hypertension": (
        "(d.icd_version = 9 AND LEFT(d.icd_code, 3) BETWEEN '401' AND '405') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) BETWEEN 'I10' AND 'I15')"
    ),
    "heart_failure": (
        "(d.icd_version = 9 AND d.icd_code LIKE '428%%') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) = 'I50')"
    ),
    "cad": (
        "(d.icd_version = 9 AND LEFT(d.icd_code, 3) BETWEEN '410' AND '414') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) BETWEEN 'I20' AND 'I25')"
    ),
    "copd": (
        "(d.icd_version = 9 AND LEFT(d.icd_code, 3) BETWEEN '490' AND '496') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 2) = 'J4')"
    ),
    "ckd": (
        "(d.icd_version = 9 AND LEFT(d.icd_code, 4) BETWEEN '5850' AND '5859') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 4) BETWEEN 'N180' AND 'N189')"
    ),
    "dyslipidemia": (
        "(d.icd_version = 9 AND d.icd_code LIKE '2720%%') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 4) = 'E780')"
    ),
    "obesity": (
        "(d.icd_version = 9 AND d.icd_code LIKE '2780%%') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) = 'E66')"
    ),
    "htg": (
        "(d.icd_version = 9 AND d.icd_code LIKE '2721%%') "
        "OR (d.icd_version = 10 AND LEFT(d.icd_code, 4) = 'E781')"
    ),
}

# 胰腺坏死 ICD
NECROSIS_ICD = (
    "(d.icd_version = 9 AND d.icd_code = '5770') AND "
    "EXISTS (SELECT 1 FROM mimiciv_hosp.diagnoses_icd d2 "
    "WHERE d2.hadm_id = d.hadm_id AND "
    "((d2.icd_version = 9 AND d2.icd_code IN ('56983','56721','56722','56729','99859','56731')) "
    "OR (d2.icd_version = 10 AND d2.icd_code IN ('K8592','K651','K659','T8143XA'))))"
)

# ── Helper: 运行查询返回字典列表 ──────────────────────────
def query(sql):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    return rows

def query_csv(filepath, sql):
    """运行查询并直接写入 CSV"""
    rows = query(sql)
    if not rows:
        print(f"  [WARN] Empty result for {filepath}")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {filepath}  ({len(rows)} rows)")

# ── Step 1: MDAP 队列定义 ─────────────────────────────────
print("=" * 60)
print("Step 1: MDAP 队列构建")
print("=" * 60)

COHORT_SQL = f"""
WITH ap_base AS (
    -- AP 诊断：ICD9 577.0 或 ICD10 K85.x
    SELECT DISTINCT d.hadm_id, d.subject_id
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE {AP_ICD_WHERE}
),
admission_info AS (
    -- 入院时间作为 T0
    SELECT a.hadm_id, a.subject_id, a.admittime AS t0,
           a.dischtime, a.deathtime, a.hospital_expire_flag,
           EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 86400.0 AS los_days,
           ROUND((EXTRACT(YEAR FROM a.admittime) - p.anchor_year) + p.anchor_age) AS age,
           p.gender
    FROM mimiciv_hosp.admissions a
    JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
    WHERE a.hadm_id IN (SELECT hadm_id FROM ap_base)
      AND ROUND((EXTRACT(YEAR FROM a.admittime) - p.anchor_year) + p.anchor_age) >= 18
      AND EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 86400.0 BETWEEN 2 AND 90
),
-- 排除病因
excluded_etiologies AS (
    SELECT DISTINCT d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE ({EXCLUDE_ICD_WHERE})
),
-- 代谢表型：TG≥500 或 肥胖(E66)/HTG(E78.1) dx
tg_ge500_anytime AS (
    SELECT DISTINCT l.hadm_id
    FROM mimiciv_hosp.labevents l
    WHERE l.itemid IN {inlist("tg")}
      AND l.valuenum >= 500
      AND l.valuenum IS NOT NULL
),
metabolic_dx AS (
    SELECT DISTINCT d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE (d.icd_version = 9 AND d.icd_code LIKE '2780%')
       OR (d.icd_version = 10 AND LEFT(d.icd_code, 3) = 'E66')
       OR (d.icd_version = 9 AND d.icd_code LIKE '2721%')
       OR (d.icd_version = 10 AND LEFT(d.icd_code, 4) = 'E781')
),
mdap_cohort AS (
    SELECT a.*
    FROM admission_info a
    WHERE a.hadm_id NOT IN (SELECT hadm_id FROM excluded_etiologies)
      AND (a.hadm_id IN (SELECT hadm_id FROM tg_ge500_anytime)
           OR a.hadm_id IN (SELECT hadm_id FROM metabolic_dx))
),
-- 入院 TG（基线表型）
admission_tg AS (
    SELECT DISTINCT ON (l.hadm_id)
           l.hadm_id, l.valuenum AS tg_admission, l.charttime AS tg_time
    FROM mimiciv_hosp.labevents l
    JOIN mdap_cohort c ON l.hadm_id = c.hadm_id
    WHERE l.itemid IN {inlist("tg")}
      AND l.valuenum IS NOT NULL
      -- 预测时点窗口与其余基线 lab 一致（-24h..+2h），避免 TG 单独享有 +6h 的不对称泄漏
      AND l.charttime BETWEEN c.t0 - INTERVAL '24 hours' AND c.t0 + INTERVAL '2 hours'
    ORDER BY l.hadm_id, ABS(EXTRACT(EPOCH FROM (l.charttime - c.t0)))
),
-- ICU 转入
icu_transfer AS (
    SELECT DISTINCT ON (c.hadm_id)
           c.hadm_id, 1 AS icu_flag,
           icu.intime AS icu_intime
    FROM mdap_cohort c
    JOIN mimiciv_icu.icustays icu ON c.hadm_id = icu.hadm_id
    ORDER BY c.hadm_id, icu.intime
),
-- 最终队列
final_cohort AS (
    SELECT c.hadm_id, c.subject_id, c.t0, c.dischtime, c.deathtime,
           c.hospital_expire_flag, c.los_days, c.age, c.gender,
           CASE WHEN tg_any.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS tg_ge500_anytime_flag,
           CASE WHEN atg.tg_admission >= 500 THEN 1 ELSE 0 END AS tg_ge500_admission_window_flag,
           CASE WHEN atg.tg_admission >= 500 THEN 1 ELSE 0 END AS tg_ge500_flag,
           CASE WHEN md.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS metabolic_dx_flag,
           atg.tg_admission, atg.tg_time,
           icu.icu_flag, icu.icu_intime
    FROM mdap_cohort c
    LEFT JOIN (SELECT DISTINCT hadm_id FROM tg_ge500_anytime) tg_any ON c.hadm_id = tg_any.hadm_id
    LEFT JOIN (SELECT DISTINCT hadm_id FROM metabolic_dx) md ON c.hadm_id = md.hadm_id
    LEFT JOIN admission_tg atg ON c.hadm_id = atg.hadm_id
    LEFT JOIN icu_transfer icu ON c.hadm_id = icu.hadm_id
)
SELECT * FROM final_cohort
ORDER BY hadm_id
"""

cohort_rows = query(COHORT_SQL)
cohort_hadm_ids = [r["hadm_id"] for r in cohort_rows]
print(f"  MDAP 队列: {len(cohort_rows)} admissions")

if len(cohort_rows) == 0:
    print("  [FATAL] 空队列，终止")
    exit(1)

# ── Step 2: 合并症 ────────────────────────────────────────
print("\nStep 2: 提取合并症")

comorbid_map = {}
for hadm_id in cohort_hadm_ids:
    comorbid_map[hadm_id] = {k: 0 for k in COMORB_WHERE}

for label, where_clause in COMORB_WHERE.items():
    sql = f"""
    SELECT DISTINCT d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE d.hadm_id = ANY(%(hadm_ids)s)
      AND ({where_clause})
    """
    cur = conn.cursor()
    cur.execute(sql, {"hadm_ids": cohort_hadm_ids})
    for (hadm_id,) in cur:
        comorbid_map[hadm_id][label] = 1
    cur.close()
    n = sum(1 for h in cohort_hadm_ids if comorbid_map[h][label] == 1)
    print(f"  {label}: {n} ({100*n/len(cohort_hadm_ids):.1f}%)")

# Charlson 简化版（基于已有 ICD 计数）
# 此处仅对已有 comorbid 做计数近似，完整 CCI 在后续单独计算
has_any = [h for h in cohort_hadm_ids if any(comorbid_map[h].values())]
print(f"  任意合并症: {len(has_any)} ({100*len(has_any)/len(cohort_hadm_ids):.1f}%)")

# ── Step 3: 基线实验室 (T0-24h ~ T0+2h) ──────────────────
print("\nStep 3: 提取基线实验室 (T0-24h ~ T0+2h)")

BASELINE_LAB_SQL = f"""
WITH cohort_t0 AS (
    SELECT hadm_id, t0 FROM (VALUES %s) AS t(hadm_id, t0)
)
SELECT l.hadm_id, l.itemid, l.valuenum,
       EXTRACT(EPOCH FROM (l.charttime - ct.t0)) / 3600.0 AS hours_from_t0
FROM mimiciv_hosp.labevents l
JOIN cohort_t0 ct ON l.hadm_id = ct.hadm_id
WHERE l.itemid IN (
    {",".join(str(i) for ids in LAB_ITEMIDS.values() for i in ids)}
)
  AND l.valuenum IS NOT NULL
  -- 防 T0 泄漏：基线特征严格限于入院即刻分诊面板（t0-24h ~ t0+2h）。
  -- 多数 ICU escalation 发生在头 6h，原 +6h 窗会让特征晚于事件。
  AND l.charttime BETWEEN ct.t0 - INTERVAL '24 hours' AND ct.t0 + INTERVAL '2 hours'
"""

# 批量查询基线实验室
t0_rows = [(r["hadm_id"], r["t0"]) for r in cohort_rows]
# 分批（PG 参数限制），每批 500
batch_size = 500
baseline_labs = {}
for i in range(0, len(t0_rows), batch_size):
    batch = t0_rows[i:i+batch_size]
    cur = conn.cursor()
    # 构建参数化查询
    placeholders = ",".join(["(%s,%s)"] * len(batch))
    flat_params = [v for pair in batch for v in pair]
    sql = BASELINE_LAB_SQL.replace("VALUES %s", f"VALUES {placeholders}")
    cur.execute(sql, flat_params)
    for row in cur:
        h = row[0]
        itemid = row[1]
        val = row[2]
        hr = row[3]
        if h not in baseline_labs:
            baseline_labs[h] = {}
        if itemid not in baseline_labs[h]:
            baseline_labs[h][itemid] = []
        baseline_labs[h][itemid].append((val, hr))
    cur.close()

# 聚合：每 marker 取最接近 T0 的一次测量
def aggregate_baseline(labs_dict, itemid_set, agg="closest"):
    result = {}
    for hadm_id, items in labs_dict.items():
        for itemid in itemid_set:
            if itemid in items and items[itemid]:
                if agg == "closest":
                    best = min(items[itemid], key=lambda x: abs(x[1]))
                    result[(hadm_id, itemid)] = best[0]
                elif agg == "first":
                    best = min(items[itemid], key=lambda x: x[1])
                    result[(hadm_id, itemid)] = best[0]
                elif agg == "mean":
                    result[(hadm_id, itemid)] = sum(v[0] for v in items[itemid]) / len(items[itemid])
    return result

# 构建 itemid -> marker 反向映射
itemid_to_marker = {}
for marker, ids in LAB_ITEMIDS.items():
    for i in ids:
        itemid_to_marker[i] = marker

def get_marker_value(lab_agg, hadm_id, marker, agg="closest"):
    """从聚合结果取 marker 的值"""
    itemids = LAB_ITEMIDS[marker]
    for itemid in itemids:
        key = (hadm_id, itemid)
        if key in lab_agg:
            return lab_agg[key]
    return None

baseline_agg = aggregate_baseline(baseline_labs, set(itemid_to_marker.keys()))

for h in cohort_hadm_ids:
    for marker in LAB_ITEMIDS:
        v = get_marker_value(baseline_agg, h, marker)
        if v is None:
            v = None  # 保持 None
        # 存储到 cohort_rows 的扩展字典
        # 后面会统一合并

print(f"  基线实验室: 覆盖 {len(baseline_labs)} / {len(cohort_hadm_ids)} 患者")

# ── Step 4: 48h 序列实验室（三窗） ────────────────────────
print("\nStep 4: 提取 48h 序列实验室 (0-6h / 6-24h / 24-48h)")

SERIAL_LAB_SQL = """
WITH cohort_t0 AS (
    SELECT hadm_id, t0 FROM (VALUES %s) AS t(hadm_id, t0)
)
SELECT l.hadm_id, l.itemid, l.valuenum,
       EXTRACT(EPOCH FROM (l.charttime - ct.t0)) / 3600.0 AS hours_from_t0
FROM mimiciv_hosp.labevents l
JOIN cohort_t0 ct ON l.hadm_id = ct.hadm_id
WHERE l.itemid IN ({itemids})
  AND l.valuenum IS NOT NULL
  AND l.charttime BETWEEN ct.t0 AND ct.t0 + INTERVAL '48 hours'
"""

# 轨迹 markers（不包含 TG，因为 G1 稀疏）
trajectory_markers = ["wbc", "lactate", "creatinine", "bun", "bilirubin", "platelet", "glucose"]
traj_itemids = []
for m in trajectory_markers:
    traj_itemids.extend(LAB_ITEMIDS[m])
traj_itemid_str = ",".join(str(i) for i in traj_itemids)

# 三窗定义
windows = [
    ("w0_6", 0, 6),
    ("w6_24", 6, 24),
    ("w24_48", 24, 48),
]

serial_data = {h: {m: {wn: [] for wn, _, _ in windows} for m in trajectory_markers} for h in cohort_hadm_ids}

for i in range(0, len(t0_rows), batch_size):
    batch = t0_rows[i:i+batch_size]
    cur = conn.cursor()
    placeholders = ",".join(["(%s,%s)"] * len(batch))
    flat_params = [v for pair in batch for v in pair]
    sql = SERIAL_LAB_SQL.format(itemids=traj_itemid_str).replace("VALUES %s", f"VALUES {placeholders}")
    cur.execute(sql, flat_params)
    for row in cur:
        h = row[0]
        itemid = row[1]
        val = row[2]
        hr = row[3]
        marker = itemid_to_marker.get(itemid)
        if marker is None:
            continue
        for wn, lo, hi in windows:
            if lo <= hr < hi:
                serial_data[h][marker][wn].append(val)
                break
    cur.close()

# 聚合：每窗取 mean
for h in cohort_hadm_ids:
    for m in trajectory_markers:
        for wn, _, _ in windows:
            vals = serial_data[h][m][wn]
            serial_data[h][m][wn] = sum(vals) / len(vals) if vals else None

# 统计覆盖率
for m in trajectory_markers:
    for wn, _, _ in windows:
        n_avail = sum(1 for h in cohort_hadm_ids if serial_data[h][m][wn] is not None)
        print(f"  {m} {wn}: {n_avail}/{len(cohort_hadm_ids)} ({100*n_avail/len(cohort_hadm_ids):.1f}%)")

# ── Step 5: 结局提取 ─────────────────────────────────────
print("\nStep 5: 结局提取")

# 28d 住院死亡
death_sql = """
SELECT c.hadm_id,
       CASE WHEN c.deathtime IS NOT NULL
            AND c.deathtime <= c.t0 + INTERVAL '28 days'
            THEN 1 ELSE 0 END AS death_28d
FROM (VALUES %s) AS c(hadm_id, t0, deathtime)
"""
death_rows = []
for i in range(0, len(t0_rows), batch_size):
    batch = [(r["hadm_id"], r["t0"], r["deathtime"]) for r in cohort_rows[i:i+batch_size]]
    cur = conn.cursor()
    placeholders = ",".join(["(%s,%s,%s)"] * len(batch))
    flat_params = [v for pair in batch for v in pair]
    sql = death_sql.replace("VALUES %s", f"VALUES {placeholders}")
    cur.execute(sql, flat_params)
    death_rows.extend([{"hadm_id": r[0], "death_28d": r[1]} for r in cur])
    cur.close()

death_map = {r["hadm_id"]: r["death_28d"] for r in death_rows}

# ICU 转入（初次 ICU 时间在 T0 之后 ≤7d）
icu_sql = f"""
SELECT DISTINCT ON (c.hadm_id)
       c.hadm_id, 1 AS icu_7d
FROM (VALUES %s) AS c(hadm_id, t0)
JOIN mimiciv_icu.icustays icu ON c.hadm_id = icu.hadm_id
WHERE icu.intime > c.t0
  AND icu.intime <= c.t0 + INTERVAL '7 days'
"""
icu_rows_list = []
for i in range(0, len(t0_rows), batch_size):
    batch = [(r["hadm_id"], r["t0"]) for r in cohort_rows[i:i+batch_size]]
    cur = conn.cursor()
    placeholders = ",".join(["(%s,%s)"] * len(batch))
    flat_params = [v for pair in batch for v in pair]
    sql = icu_sql.replace("VALUES %s", f"VALUES {placeholders}")
    cur.execute(sql, flat_params)
    icu_rows_list.extend([{"hadm_id": r[0], "icu_7d": r[1]} for r in cur])
    cur.close()

icu_7d_map = {r["hadm_id"]: r["icu_7d"] for r in icu_rows_list}

# 住院死亡
hosp_death = sum(1 for r in cohort_rows if r["hospital_expire_flag"] == 1)
print(f"  住院死亡: {hosp_death} ({100*hosp_death/len(cohort_rows):.1f}%)")
print(f"  28d 死亡: {sum(death_map.values())}")
print(f"  7d ICU 转入: {sum(icu_7d_map.values())}")

# ── Step 6: 组装 Canonical 宽表 ───────────────────────────
print("\nStep 6: 组装 Canonical MDAP 宽表")

COLUMNS = [
    "hadm_id", "subject_id", "age", "gender",
    # 代谢表型
    "tg_ge500_anytime_flag", "tg_ge500_admission_window_flag",
    "tg_ge500_flag", "metabolic_dx_flag", "tg_admission",
    # 合并症
    "diabetes", "hypertension", "heart_failure", "cad", "copd", "ckd",
    "dyslipidemia", "obesity_dx", "htg_dx",
    # 基线实验室（最接近 T0）
    "baseline_wbc", "baseline_lactate", "baseline_creatinine", "baseline_bun",
    "baseline_bilirubin", "baseline_platelet", "baseline_glucose",
    "baseline_lipase", "baseline_amylase", "baseline_calcium",
    "baseline_alt", "baseline_ast", "baseline_tg",
    # 轨迹实验室（三窗 × 7 marker）
]

for m in trajectory_markers:
    for wn, _, _ in windows:
        COLUMNS.append(f"{m}_{wn}")

COLUMNS.extend([
    # 结局
    "icu_flag", "icu_7d", "death_28d", "hospital_expire_flag",
    "los_days",
    # 地标-相对时间（用于 at-risk / landmark-relative outcome，防泄漏）
    "icu_intime_hours", "death_offset_hours",
    # T0
    "t0",
])

canonical_rows = []
for r in cohort_rows:
    h = r["hadm_id"]
    row = {
        "hadm_id": h,
        "subject_id": r["subject_id"],
        "age": int(r["age"]) if r["age"] else None,
        "gender": r["gender"],
        "tg_ge500_anytime_flag": int(r["tg_ge500_anytime_flag"] or 0),
        "tg_ge500_admission_window_flag": int(r["tg_ge500_admission_window_flag"] or 0),
        "tg_ge500_flag": int(r["tg_ge500_flag"] or 0),
        "metabolic_dx_flag": int(r["metabolic_dx_flag"] or 0),
        "tg_admission": r["tg_admission"],
        "diabetes": comorbid_map[h]["diabetes"],
        "hypertension": comorbid_map[h]["hypertension"],
        "heart_failure": comorbid_map[h]["heart_failure"],
        "cad": comorbid_map[h]["cad"],
        "copd": comorbid_map[h]["copd"],
        "ckd": comorbid_map[h]["ckd"],
        "dyslipidemia": comorbid_map[h]["dyslipidemia"],
        "obesity_dx": comorbid_map[h]["obesity"],
        "htg_dx": comorbid_map[h]["htg"],
    }
    # 基线 labs
    for marker in LAB_ITEMIDS:
        val = get_marker_value(baseline_agg, h, marker)
        row[f"baseline_{marker}"] = round(val, 2) if val is not None else None
    # 轨迹
    for m in trajectory_markers:
        for wn, _, _ in windows:
            val = serial_data[h][m][wn]
            row[f"{m}_{wn}"] = round(val, 2) if val is not None else None
    # 结局
    row["icu_flag"] = r["icu_flag"] or 0
    row["icu_7d"] = icu_7d_map.get(h, 0)
    row["death_28d"] = death_map.get(h, 0)
    row["hospital_expire_flag"] = r["hospital_expire_flag"]
    row["los_days"] = round(r["los_days"], 1)
    # 地标-相对时间（小时）：ICU 首次转入、死亡相对 T0
    icu_it = r["icu_intime"]
    row["icu_intime_hours"] = (
        round((icu_it - r["t0"]).total_seconds() / 3600.0, 2) if icu_it else None
    )
    dth = r["deathtime"]
    row["death_offset_hours"] = (
        round((dth - r["t0"]).total_seconds() / 3600.0, 2) if dth else None
    )
    row["t0"] = str(r["t0"])
    canonical_rows.append(row)

# 写 CSV
canonical_path = os.path.join(OUTDIR, "canonical_mdap_cohort.csv")
with open(canonical_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(canonical_rows)
print(f"  -> {canonical_path}  ({len(canonical_rows)} rows, {len(COLUMNS)} cols)")

# ── Step 7: Table 1 汇总 ──────────────────────────────────
print("\nStep 7: Table 1 汇总")

def _linear_quantile(vals, q):
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac

def _format_num(v, decimals):
    if decimals == 0:
        return str(int(round(v)))
    return f"{round(v, decimals):.{decimals}f}"

def decimals_for(key):
    zero_decimal = {
        "age", "tg_admission", "baseline_platelet", "baseline_glucose",
        "baseline_lipase", "baseline_amylase", "baseline_alt", "baseline_ast",
    }
    return 0 if key in zero_decimal else 1

def median_iqr(vals, key=None):
    vals = sorted([float(v) for v in vals if v is not None])
    if not vals:
        return None, None, None
    decimals = decimals_for(key) if key else 1
    q1 = _linear_quantile(vals, 0.25)
    q2 = _linear_quantile(vals, 0.50)
    q3 = _linear_quantile(vals, 0.75)
    return _format_num(q2, decimals), _format_num(q1, decimals), _format_num(q3, decimals)

def pct_n(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    s = sum(1 for v in vals if v == 1)
    return s, n, round(100 * s / n, 1) if n else None

table1_rows = []
for label, key, vtype in [
    ("N", "hadm_id", "count"),
    ("Age, median (IQR)", "age", "cont"),
    ("Male, n (%)", "gender", "cat_male"),
    # 代谢表型
    ("Any-time TG >=500 mg/dL, n (%)", "tg_ge500_anytime_flag", "cat"),
    ("Admission-window TG >=500 mg/dL, n (%)", "tg_ge500_admission_window_flag", "cat"),
    ("Metabolic dx (obesity/HTG), n (%)", "metabolic_dx_flag", "cat"),
    ("TG admission, median (IQR)", "tg_admission", "cont"),
    # 合并症
    ("Diabetes, n (%)", "diabetes", "cat"),
    ("Hypertension, n (%)", "hypertension", "cat"),
    ("Heart failure, n (%)", "heart_failure", "cat"),
    ("CAD, n (%)", "cad", "cat"),
    ("COPD, n (%)", "copd", "cat"),
    ("CKD, n (%)", "ckd", "cat"),
    ("Dyslipidemia, n (%)", "dyslipidemia", "cat"),
    ("Obesity dx, n (%)", "obesity_dx", "cat"),
    ("HTG dx, n (%)", "htg_dx", "cat"),
    # 基线实验室
    ("Baseline WBC, median (IQR)", "baseline_wbc", "cont"),
    ("Baseline Lactate, median (IQR)", "baseline_lactate", "cont"),
    ("Baseline Creatinine, median (IQR)", "baseline_creatinine", "cont"),
    ("Baseline BUN, median (IQR)", "baseline_bun", "cont"),
    ("Baseline Bilirubin, median (IQR)", "baseline_bilirubin", "cont"),
    ("Baseline Platelet, median (IQR)", "baseline_platelet", "cont"),
    ("Baseline Glucose, median (IQR)", "baseline_glucose", "cont"),
    ("Baseline Lipase, median (IQR)", "baseline_lipase", "cont"),
    ("Baseline Amylase, median (IQR)", "baseline_amylase", "cont"),
    ("Baseline Calcium, median (IQR)", "baseline_calcium", "cont"),
    ("Baseline ALT, median (IQR)", "baseline_alt", "cont"),
    ("Baseline AST, median (IQR)", "baseline_ast", "cont"),
    # 结局
    ("ICU anytime, n (%)", "icu_flag", "cat"),
    ("ICU within 7d, n (%)", "icu_7d", "cat"),
    ("28-day mortality, n (%)", "death_28d", "cat"),
    ("Hospital mortality, n (%)", "hospital_expire_flag", "cat"),
    ("LOS days, median (IQR)", "los_days", "cont"),
]:
    vals = [row[key] for row in canonical_rows]
    if vtype == "count":
        stat = str(len(vals))
    elif vtype == "cont":
        m, q1, q3 = median_iqr(vals, key)
        stat = f"{m} ({q1}-{q3})" if m is not None else "N/A"
    elif vtype == "cat":
        s, n, p = pct_n(vals)
        stat = f"{s} ({p}%)" if n else "N/A"
    elif vtype == "cat_male":
        s = sum(1 for v in vals if v == "M")
        p = round(100 * s / len(vals), 1)
        stat = f"{s} ({p}%)"
    table1_rows.append({"Variable": label, "Overall (N=" + str(len(canonical_rows)) + ")": stat})

# 亚组：TG≥500 vs TG<500/missing
tg_pos = [row for row in canonical_rows if row["tg_ge500_anytime_flag"] == 1]
tg_neg = [row for row in canonical_rows if row["tg_ge500_anytime_flag"] == 0]
# 复用主循环的变量定义
subgroup_vars = [
    ("Age, median (IQR)", "age", "cont"),
    ("Male, n (%)", "gender", "cat_male"),
    ("Any-time TG >=500 mg/dL, n (%)", "tg_ge500_anytime_flag", "cat"),
    ("Admission-window TG >=500 mg/dL, n (%)", "tg_ge500_admission_window_flag", "cat"),
    ("Metabolic dx (obesity/HTG), n (%)", "metabolic_dx_flag", "cat"),
    ("TG admission, median (IQR)", "tg_admission", "cont"),
    ("Diabetes, n (%)", "diabetes", "cat"),
    ("Hypertension, n (%)", "hypertension", "cat"),
    ("Heart failure, n (%)", "heart_failure", "cat"),
    ("CAD, n (%)", "cad", "cat"),
    ("COPD, n (%)", "copd", "cat"),
    ("CKD, n (%)", "ckd", "cat"),
    ("Dyslipidemia, n (%)", "dyslipidemia", "cat"),
    ("Baseline WBC, median (IQR)", "baseline_wbc", "cont"),
    ("Baseline Lactate, median (IQR)", "baseline_lactate", "cont"),
    ("Baseline Creatinine, median (IQR)", "baseline_creatinine", "cont"),
    ("Baseline BUN, median (IQR)", "baseline_bun", "cont"),
    ("Baseline Bilirubin, median (IQR)", "baseline_bilirubin", "cont"),
    ("Baseline Platelet, median (IQR)", "baseline_platelet", "cont"),
    ("Baseline Glucose, median (IQR)", "baseline_glucose", "cont"),
    ("Baseline Lipase, median (IQR)", "baseline_lipase", "cont"),
    ("ICU anytime, n (%)", "icu_flag", "cat"),
    ("ICU within 7d, n (%)", "icu_7d", "cat"),
    ("28-day mortality, n (%)", "death_28d", "cat"),
    ("Hospital mortality, n (%)", "hospital_expire_flag", "cat"),
    ("LOS days, median (IQR)", "los_days", "cont"),
]
for label, key, vtype in subgroup_vars:
    vals_pos = [row[key] for row in tg_pos]
    vals_neg = [row[key] for row in tg_neg]
    if vtype == "cont":
        m, q1, q3 = median_iqr(vals_pos, key)
        stat_pos = f"{m} ({q1}-{q3})" if m is not None else "N/A"
        m2, q1b, q3b = median_iqr(vals_neg, key)
        stat_neg = f"{m2} ({q1b}-{q3b})" if m2 is not None else "N/A"
    elif vtype == "cat":
        s_p, n_p, p_p = pct_n(vals_pos)
        stat_pos = f"{s_p} ({p_p}%)" if n_p else "N/A"
        s_n, n_n, p_n = pct_n(vals_neg)
        stat_neg = f"{s_n} ({p_n}%)" if n_n else "N/A"
    elif vtype == "cat_male":
        s_p = sum(1 for v in vals_pos if v == "M")
        p_p = round(100 * s_p / len(vals_pos), 1) if vals_pos else None
        stat_pos = f"{s_p} ({p_p}%)" if vals_pos else "N/A"
        s_n = sum(1 for v in vals_neg if v == "M")
        p_n = round(100 * s_n / len(vals_neg), 1) if vals_neg else None
        stat_neg = f"{s_n} ({p_n}%)" if vals_neg else "N/A"
    else:
        continue
    for tr in table1_rows:
        if tr["Variable"] == label:
            tr[f"Any-time TG>=500 (N={len(tg_pos)})"] = stat_pos
            tr[f"Other MDAP (N={len(tg_neg)})"] = stat_neg
            break

# 收集所有列名（含亚组列）
fieldnames_t1 = []
for row in table1_rows:
    for k in row:
        if k not in fieldnames_t1:
            fieldnames_t1.append(k)
t1_path = os.path.join(OUTDIR, "table1_summary.csv")
with open(t1_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames_t1)
    w.writeheader()
    w.writerows(table1_rows)
print(f"  -> {t1_path}  ({len(table1_rows)} variables)")

# ── 完成 ─────────────────────────────────────────────────
conn.close()
print(f"\n{'=' * 60}")
print(f"Done. 输出目录: {OUTDIR}")
print(f"  canonical_mdap_cohort.csv : {len(canonical_rows)} hadm_ids, {len(COLUMNS)} columns")
print(f"  table1_summary.csv        : {len(table1_rows)} variables")
print(f"{'=' * 60}")
