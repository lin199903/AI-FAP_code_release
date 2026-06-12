"""
P0 reviewer check: the corrected admission-window TG flag/continuous value was
extracted with -24h..+6h, while every OTHER baseline lab uses -24h..+2h.
This re-extracts the nearest-to-T0 TG within -24h..+2h and re-runs the EXACT
repeated-CV/OOF pipeline of 06_robust_cv_oof.py to test whether the headline
OOF AUROC (0.743, +6h) reproduces under the unified -24h..+2h window.
"""
import os, numpy as np, pandas as pd, warnings, psycopg2
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))
from _dbconfig import get_dsn
DSN = get_dsn()

# ---- re-extract nearest TG within -24h..+2h ----
conn = psycopg2.connect(DSN)
rows = list(df[["hadm_id", "t0"]].itertuples(index=False, name=None))
cur = conn.cursor()
from psycopg2.extras import execute_values
cur.execute("CREATE TEMP TABLE _c (hadm_id bigint, t0 timestamp)")
execute_values(cur, "INSERT INTO _c VALUES %s", rows)
cur.execute("""
    SELECT DISTINCT ON (l.hadm_id) l.hadm_id, l.valuenum
    FROM mimiciv_hosp.labevents l JOIN _c c ON l.hadm_id = c.hadm_id
    WHERE l.itemid = 51000 AND l.valuenum IS NOT NULL
      AND l.charttime BETWEEN c.t0 - INTERVAL '24 hours' AND c.t0 + INTERVAL '2 hours'
    ORDER BY l.hadm_id, ABS(EXTRACT(EPOCH FROM (l.charttime - c.t0)))
""")
tg_p2h = {h: v for h, v in cur.fetchall()}
conn.close()
df["tg_admission_p2h"] = df["hadm_id"].map(tg_p2h)

tg6 = pd.to_numeric(df["tg_admission"], errors="coerce")
tg2 = pd.to_numeric(df["tg_admission_p2h"], errors="coerce")
print("=== TG measurement window comparison ===")
print(f"  measured in -24/+6h : {tg6.notna().sum():3d}   >=500: {(tg6>=500).sum()}")
print(f"  measured in -24/+2h : {tg2.notna().sum():3d}   >=500: {(tg2>=500).sum()}")
print(f"  rows that LOSE a TG value when tightening +6h->+2h: {(tg6.notna() & tg2.isna()).sum()}")
print(f"  rows whose continuous TG value CHANGES: {(tg6.fillna(-1).round(3) != tg2.fillna(-1).round(3)).sum()}")
print(f"  flag (>=500) changes: {((tg6>=500).astype(int) != (tg2>=500).astype(int)).sum()}")

# ---- EXACT 06 pipeline ----
icu_h, dth_h = df["icu_intime_hours"], df["death_offset_hours"]
y = (((icu_h.notna()) & (icu_h > 0) & (icu_h <= 168)) |
     ((df["death_28d"] == 1) & dth_h.notna() & (dth_h > 0))).astype(int).values
FEATS = ["age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission", "diabetes", "hypertension",
         "heart_failure", "cad", "copd", "ckd", "dyslipidemia", "obesity_dx", "htg_dx",
         "baseline_wbc", "baseline_creatinine", "baseline_bun", "baseline_bilirubin",
         "baseline_platelet", "baseline_glucose", "baseline_lipase", "baseline_calcium"]
LGB = dict(objective="binary", verbosity=-1, n_estimators=200, max_depth=4, num_leaves=15,
           learning_rate=0.05, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
           reg_alpha=0.1, reg_lambda=1.0, random_state=42, is_unbalance=False)

def fit_fold(Xtr, ytr, Xte):
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)
    Xtr2 = np.where(np.isnan(Xtr), med, Xtr); Xte2 = np.where(np.isnan(Xte), med, Xte)
    return lgb.LGBMClassifier(**LGB).fit(Xtr2, ytr).predict_proba(Xte2)[:, 1]

def repeated_oof(X, yv, folds=10, repeats=10):
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=2024)
    cur = np.zeros(len(yv)); per_rep = []; k = 0
    for tr, te in rskf.split(X, yv):
        cur[te] = fit_fold(X[tr], yv[tr], X[te]); k += 1
        if k == folds:
            per_rep.append(roc_auc_score(yv, cur)); cur = np.zeros(len(yv)); k = 0
    return np.array(per_rep)

def Xmat(tg_cont, flag):
    d = df.copy(); d["tg_admission"] = tg_cont.values; d["tg_ge500_flag"] = flag
    return d[FEATS].astype(float).values

for tag, cont in [("+6h (published)", tg6), ("+2h (unified)", tg2)]:
    X = Xmat(cont, (cont >= 500).astype(int).values)
    a = repeated_oof(X, y)
    ci = np.percentile(a, [2.5, 97.5])
    print(f"  OOF AUROC {tag:18s}: {a.mean():.3f}  (repeat 2.5-97.5%: {ci[0]:.3f}-{ci[1]:.3f})")
