"""
Are the two governance REVERSALS (abstention paradox, isotonic recalibration benefit)
robust, or LightGBM-specific artifacts? Replicate 04's exact abstention + recalibration
logic for {LEAKY vs CORRECTED} x {several models/seeds}.

Abstention paradox  := withheld (least-confident, |p-0.5| smallest) event rate > overall prevalence.
Isotonic 'benefit'  := isotonic ECE < raw ECE (improves). We expect: LEAKY shows benefit+paradox,
CORRECTED shows neither, if the reversal is real.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))
icu_h, dth_h = df["icu_intime_hours"], df["death_offset_hours"]
df["_y"] = (((icu_h.notna()) & (icu_h > 0) & (icu_h <= 168)) | ((df["death_28d"] == 1) & dth_h.notna() & (dth_h > 0))).astype(int)
df["t0_dt"] = pd.to_datetime(df["t0"]); df = df.sort_values("t0_dt").reset_index(drop=True)
split = int(len(df) * 0.7)
tr = df["hadm_id"].isin(set(df.iloc[:split]["hadm_id"])); va = df["hadm_id"].isin(set(df.iloc[split:]["hadm_id"]))
FEATS = ["age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission", "diabetes", "hypertension",
         "heart_failure", "cad", "copd", "ckd", "dyslipidemia", "obesity_dx", "htg_dx",
         "baseline_wbc", "baseline_creatinine", "baseline_bun", "baseline_bilirubin",
         "baseline_platelet", "baseline_glucose", "baseline_lipase", "baseline_calcium"]
corrected = (pd.to_numeric(df["tg_admission"], errors="coerce") >= 500).astype(int).values
leaky = df["tg_ge500_anytime_flag"].astype(int).values

def ece(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1])
        if m.sum(): e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return e

def Xy(fv):
    d = df.copy(); d["tg_ge500_flag"] = fv
    Xtr, Xva = d.loc[tr, FEATS].copy(), d.loc[va, FEATS].copy()
    for c in FEATS:
        m = Xtr[c].median(); m = 0 if pd.isna(m) else m
        Xtr[c] = Xtr[c].fillna(m); Xva[c] = Xva[c].fillna(m)
    return Xtr, Xva, d.loc[tr, "_y"].values, d.loc[va, "_y"].values

def fit_predict(kind, Xtr, Xva, ytr, seed=42):
    if kind == "lgbm":
        m = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_estimators=200, max_depth=4,
            num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed, is_unbalance=False).fit(Xtr, ytr)
        return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xva)[:, 1]
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=400, random_state=seed).fit(Xtr, ytr)
        return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xva)[:, 1]
    sc = StandardScaler().fit(Xtr); a, b = sc.transform(Xtr), sc.transform(Xva)
    if kind == "logit": m = LogisticRegression(max_iter=5000)
    else: m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000)
    m.fit(a, ytr); return m.predict_proba(a)[:, 1], m.predict_proba(b)[:, 1]

def analyse(ptr, pva, ytr, yva):
    prev = yva.mean()
    raw_ece = ece(yva, pva)
    ir = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr)
    iso_ece = ece(yva, ir.transform(pva))
    iso_improves = iso_ece < raw_ece
    conf = np.maximum(pva, 1 - pva); order = np.argsort(conf)
    res = {}
    for pct in (20, 30):
        n = int(len(yva) * pct / 100); wr = yva[order[:n]].mean()
        res[pct] = wr
    return raw_ece, iso_ece, iso_improves, res, prev

MODELS = [("lgbm", 42), ("lgbm", 1), ("lgbm", 7), ("logit", 42), ("enet", 42), ("rf", 42)]
for tag, fv in [("LEAKY (any-time)", leaky), ("CORRECTED (adm-window)", corrected)]:
    Xtr, Xva, ytr, yva = Xy(fv)
    print("\n" + "=" * 96)
    print(f"{tag}   (val prevalence = {yva.mean():.3f})")
    print(f"{'model':14s} {'rawECE':>7s} {'isoECE':>7s} {'iso?':>9s}   {'withheld@20%':>13s} {'withheld@30%':>13s}   paradox?")
    for kind, seed in MODELS:
        ptr, pva = fit_predict(kind, Xtr, Xva, ytr, seed)
        r, iso, impr, wr, prev = analyse(ptr, pva, ytr, yva)
        para20 = wr[20] > prev; para30 = wr[30] > prev
        name = f"{kind}/{seed}"
        print(f"{name:14s} {r:7.3f} {iso:7.3f} {('IMPROVES' if impr else 'worsens'):>9s}   "
              f"{wr[20]:.3f}{'*' if para20 else ' ':>1s}{'':>6s} {wr[30]:.3f}{'*' if para30 else ' ':>1s}{'':>6s}   "
              f"{'YES' if (para20 or para30) else 'no'}")
    print(f"  (* = withheld event rate exceeds overall prevalence {yva.mean():.3f} -> abstention paradox)")
