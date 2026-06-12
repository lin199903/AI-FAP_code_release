"""
AUDIT RE-RUN: quantify the impact of the leaky any-time TG>=500 flag on the
T0 admission model. Faithfully reproduces 03_landmark_ml.py's T0 pipeline
(same temporal split, same LightGBM params, same median imputation, same
1000-bootstrap CI) and swaps ONLY the definition of the triglyceride flag.

Variants:
  A  original  : tg_ge500_flag = any TG>=500 ANYTIME during admission (LEAKY, as published)
  B  corrected : tg_ge500_flag = admission-window TG>=500 (-24h..+6h nearest T0); missing -> 0
  C  dropped   : remove the TG>=500 flag entirely (keep tg_admission continuous)
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import shap

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUTDIR, "canonical_mdap_cohort.csv"))

# ---- composite outcome + landmark-relative T0 outcome (identical to source) ----
df["composite_outcome"] = ((df["icu_7d"] == 1) | (df["hospital_expire_flag"] == 1)).astype(int)
ICU_HORIZON_H = 168
icu_h, dth_h = df["icu_intime_hours"], df["death_offset_hours"]
icu_after = icu_h.notna() & (icu_h > 0) & (icu_h <= ICU_HORIZON_H)
death_after = (df["death_28d"] == 1) & dth_h.notna() & (dth_h > 0)
df["_y"] = (icu_after | death_after).astype(int)

# ---- temporal 70/30 split (identical) ----
df["t0_dt"] = pd.to_datetime(df["t0"])
df = df.sort_values("t0_dt").reset_index(drop=True)
split_idx = int(len(df) * 0.7)
train_hadm = set(df.iloc[:split_idx]["hadm_id"]); val_hadm = set(df.iloc[split_idx:]["hadm_id"])

BASELINE_FEATURES = [
    "age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission",
    "diabetes", "hypertension", "heart_failure", "cad", "copd", "ckd",
    "dyslipidemia", "obesity_dx", "htg_dx",
    "baseline_wbc", "baseline_creatinine", "baseline_bun",
    "baseline_bilirubin", "baseline_platelet", "baseline_glucose",
    "baseline_lipase", "baseline_calcium",
]
LGB_PARAMS = dict(objective="binary", metric="auc", verbosity=-1, n_estimators=200,
                  max_depth=4, num_leaves=15, learning_rate=0.05, min_child_samples=20,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=42, is_unbalance=False)

def compute_ece(y, p, n_bins=10):
    b = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p >= b[i]) & (p < b[i + 1])
        if m.sum(): e += m.sum()/len(p)*abs(y[m].mean()-p[m].mean())
    return e

def cal_slope(y, p):
    eps = 1e-6; pp = np.clip(p, eps, 1-eps); lg = np.log(pp/(1-pp))
    lr = LogisticRegression(fit_intercept=True, max_iter=1000).fit(lg.reshape(-1,1), y)
    return lr.intercept_[0], lr.coef_[0][0]

def run(tag, frame, feats):
    d = frame.copy()
    tr, va = d["hadm_id"].isin(train_hadm), d["hadm_id"].isin(val_hadm)
    Xtr, Xva = d.loc[tr, feats].copy(), d.loc[va, feats].copy()
    for c in feats:
        med = Xtr[c].median();  med = 0 if pd.isna(med) else med
        Xtr[c] = Xtr[c].fillna(med); Xva[c] = Xva[c].fillna(med)
    ytr, yva = d.loc[tr, "_y"].values, d.loc[va, "_y"].values
    m = lgb.LGBMClassifier(**LGB_PARAMS).fit(Xtr, ytr)
    pv = m.predict_proba(Xva)[:, 1]
    auroc = roc_auc_score(yva, pv)
    # bootstrap CI (identical seed/N)
    np.random.seed(42); boots = []
    for _ in range(1000):
        idx = np.random.choice(len(yva), len(yva), replace=True)
        if yva[idx].sum() == 0 or (1-yva[idx]).sum() == 0: continue
        boots.append(roc_auc_score(yva[idx], pv[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ci, sl = cal_slope(yva, pv)
    # SHAP top-5
    sv = shap.TreeExplainer(m).shap_values(Xva)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
    print(f"\n### {tag}")
    print(f"  n_features={len(feats)}  val n={len(yva)} events={int(yva.sum())}")
    print(f"  AUROC={auroc:.3f} (95% CI {lo:.3f}-{hi:.3f})  AUPRC={average_precision_score(yva,pv):.3f}"
          f"  Brier={brier_score_loss(yva,pv):.3f}  ECE={compute_ece(yva,pv):.3f}  cal_slope={sl:.3f}")
    print(f"  SHAP top-5: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.head(5).items()))
    return auroc, (lo, hi)

# ---- prevalence of each flag definition ----
n_any = int((df["tg_ge500_flag"] == 1).sum())
adm_flag = (pd.to_numeric(df["tg_admission"], errors="coerce") >= 500).astype(int)
print(f"Flag prevalence:  any-time TG>=500 (published) = {n_any}/{len(df)}"
      f"   admission-window TG>=500 = {int(adm_flag.sum())}/{len(df)}")
print(f"Admission TG measured (-24h..+6h) = {int(pd.to_numeric(df['tg_admission'],errors='coerce').notna().sum())}/{len(df)}")

# Variant A: as published
run("A  ORIGINAL  (leaky any-time TG>=500 flag)", df, BASELINE_FEATURES)

# Variant B: corrected admission-window flag
dfB = df.copy(); dfB["tg_ge500_flag"] = adm_flag.values
run("B  CORRECTED (admission-window TG>=500 flag)", dfB, BASELINE_FEATURES)

# Variant C: drop the flag entirely
featsC = [f for f in BASELINE_FEATURES if f != "tg_ge500_flag"]
run("C  DROPPED   (no TG>=500 flag; keep continuous tg_admission)", df, featsC)
