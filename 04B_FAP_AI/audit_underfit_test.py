"""
Does the corrected (lower) AUROC reflect UNDER-FITTING rather than removed leakage?
Test: sweep model capacity/family for BOTH the leaky any-time flag and the corrected
admission-window flag. If higher capacity closes the gap -> under-fitting. If the leaky
flag stays ~+0.06 at every capacity AND train>>val -> the flag adds (leaky) signal.
Also report train vs val AUROC (over/under-fit diagnosis) and multi-seed spread.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))  # current exported cohort stores both corrected and any-time TG flags
icu_h, dth_h = df["icu_intime_hours"], df["death_offset_hours"]
df["_y"] = (((icu_h.notna()) & (icu_h > 0) & (icu_h <= 168)) | ((df["death_28d"] == 1) & dth_h.notna() & (dth_h > 0))).astype(int)
df["t0_dt"] = pd.to_datetime(df["t0"]); df = df.sort_values("t0_dt").reset_index(drop=True)
split = int(len(df) * 0.7)
tr = df["hadm_id"].isin(set(df.iloc[:split]["hadm_id"])); va = df["hadm_id"].isin(set(df.iloc[split:]["hadm_id"]))

FEATS = ["age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission", "diabetes", "hypertension",
         "heart_failure", "cad", "copd", "ckd", "dyslipidemia", "obesity_dx", "htg_dx",
         "baseline_wbc", "baseline_creatinine", "baseline_bun", "baseline_bilirubin",
         "baseline_platelet", "baseline_glucose", "baseline_lipase", "baseline_calcium"]

# build leaky vs corrected flag versions
corrected_flag = (pd.to_numeric(df["tg_admission"], errors="coerce") >= 500).astype(int).values
# reconstruct leaky any-time flag from the current canonical cohort
leaky = df["tg_ge500_anytime_flag"].astype(int).values

def Xy(flagvals):
    d = df.copy(); d["tg_ge500_flag"] = flagvals
    Xtr, Xva = d.loc[tr, FEATS].copy(), d.loc[va, FEATS].copy()
    for c in FEATS:
        m = Xtr[c].median(); m = 0 if pd.isna(m) else m
        Xtr[c] = Xtr[c].fillna(m); Xva[c] = Xva[c].fillna(m)
    return Xtr, Xva, d.loc[tr, "_y"].values, d.loc[va, "_y"].values

def lgbm(**kw):
    base = dict(objective="binary", verbosity=-1, random_state=42, is_unbalance=False)
    base.update(kw); return lgb.LGBMClassifier(**base)

CONFIGS = {
    "orig (d4,l15,n200,reg)": dict(n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
                                   min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
    "deeper (d6,l31,n400)":   dict(n_estimators=400, max_depth=6, num_leaves=31, learning_rate=0.05, min_child_samples=10),
    "very deep (d-1,l63,n800)": dict(n_estimators=800, max_depth=-1, num_leaves=63, learning_rate=0.03, min_child_samples=5),
    "lgbm default":           dict(),
}
print("="*78)
print("CAPACITY SWEEP — validation AUROC (train AUROC in parens)")
print(f"{'config':30s} {'LEAKY flag':>22s} {'CORRECTED flag':>22s}   gap")
for name, kw in CONFIGS.items():
    row = []
    for fv in (leaky, corrected_flag):
        Xtr, Xva, ytr, yva = Xy(fv)
        m = lgbm(**kw).fit(Xtr, ytr)
        atr = roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])
        ava = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        row.append((ava, atr))
    gap = row[0][0] - row[1][0]
    print(f"{name:30s}  {row[0][0]:.3f} (tr {row[0][1]:.3f})   {row[1][0]:.3f} (tr {row[1][1]:.3f})   {gap:+.3f}")

print("\n" + "="*78)
print("OTHER MODEL FAMILIES (corrected flag) — val AUROC")
Xtr, Xva, ytr, yva = Xy(corrected_flag)
sc = StandardScaler().fit(Xtr)
for name, mdl in [
    ("Logistic (L2)", LogisticRegression(max_iter=5000)),
    ("ElasticNet logistic", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000)),
    ("RandomForest(400)", RandomForestClassifier(n_estimators=400, random_state=42)),
]:
    Xt = sc.transform(Xtr) if "orest" not in name else Xtr.values
    Xv = sc.transform(Xva) if "orest" not in name else Xva.values
    mdl.fit(Xt, ytr); av = roc_auc_score(yva, mdl.predict_proba(Xv)[:, 1])
    print(f"  {name:24s} val AUROC = {av:.3f}")

print("\n" + "="*78)
print("MULTI-SEED spread of CORRECTED model (orig params), seeds 0..19")
Xtr, Xva, ytr, yva = Xy(corrected_flag)
aucs = []
for s in range(20):
    m = lgbm(n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05, min_child_samples=20,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=s,
             bagging_seed=s, feature_fraction_seed=s).fit(Xtr, ytr)
    aucs.append(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
print(f"  corrected val AUROC: mean={np.mean(aucs):.3f}  sd={np.std(aucs):.3f}  min={min(aucs):.3f}  max={max(aucs):.3f}")
