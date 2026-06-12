"""
P0-1 fix: unify the admission TG window to -24h..+2h (matching every other
baseline lab). Re-extract nearest-to-T0 TG within -24/+2h from MIMIC and patch
the canonical cohort's tg_admission (continuous) and admission-window TG flags.
The any-time TG flag used for descriptive Table 1 stratification is preserved.
"""
import os, shutil, pandas as pd, psycopg2
from psycopg2.extras import execute_values

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CANON = os.path.join(OUT, "canonical_mdap_cohort.csv")
from _dbconfig import get_dsn
DSN = get_dsn()

df = pd.read_csv(CANON)
shutil.copy(CANON, os.path.join(OUT, "canonical_mdap_cohort_pre_pm2h_backup.csv"))

conn = psycopg2.connect(DSN); cur = conn.cursor()
cur.execute("CREATE TEMP TABLE _c (hadm_id bigint, t0 timestamp)")
execute_values(cur, "INSERT INTO _c VALUES %s",
               list(df[["hadm_id", "t0"]].itertuples(index=False, name=None)))
cur.execute("""
    SELECT DISTINCT ON (l.hadm_id) l.hadm_id, l.valuenum
    FROM mimiciv_hosp.labevents l JOIN _c c ON l.hadm_id = c.hadm_id
    WHERE l.itemid = 51000 AND l.valuenum IS NOT NULL
      AND l.charttime BETWEEN c.t0 - INTERVAL '24 hours' AND c.t0 + INTERVAL '2 hours'
    ORDER BY l.hadm_id, ABS(EXTRACT(EPOCH FROM (l.charttime - c.t0)))
""")
tg_p2h = {h: v for h, v in cur.fetchall()}
conn.close()

old_meas = pd.to_numeric(df["tg_admission"], errors="coerce").notna().sum()
old_flag = int(df.get("tg_ge500_admission_window_flag", df["tg_ge500_flag"]).sum())
df["tg_admission"] = df["hadm_id"].map(tg_p2h)
tg = pd.to_numeric(df["tg_admission"], errors="coerce")
df["tg_ge500_admission_window_flag"] = (tg >= 500).astype(int)
# Backward-compatible alias retained for older modelling scripts.
df["tg_ge500_flag"] = df["tg_ge500_admission_window_flag"]
# baseline_tg mirrors the admission TG continuous value where present
if "baseline_tg" in df.columns:
    df["baseline_tg"] = df["tg_admission"]
df.to_csv(CANON, index=False)
print(f"canonical patched to -24/+2h TG.")
print(f"  tg_admission measured: {old_meas} (+6h) -> {int(tg.notna().sum())} (+2h)")
print(f"  tg_ge500_admission_window_flag: {old_flag} (+6h) -> {int(df['tg_ge500_admission_window_flag'].sum())} (+2h)")
if "tg_ge500_anytime_flag" in df.columns:
    print(f"  tg_ge500_anytime_flag preserved: {int(df['tg_ge500_anytime_flag'].sum())}")
print(f"  cohort n unchanged: {len(df)}")
