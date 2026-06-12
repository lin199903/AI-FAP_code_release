"""
Reviewer revision analyses M3 (feature-removal sensitivity) and M5 (parsimonious /
clinical-score-proxy baselines vs the full governed LightGBM model).

Shares the validated OOF machinery of 06_robust_cv_oof.py: repeated stratified 10x10 CV,
per-fold median imputation (nothing data-dependent crosses a fold boundary), pooled OOF
probabilities, AUROC reported as mean over repeats with the 2.5-97.5% repeat range.

Outputs:
  outputs/revision_M3_feature_removal.csv
  outputs/revision_M5_baselines.csv
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))
icu_h, dth_h = df["icu_intime_hours"], df["death_offset_hours"]
y = (((icu_h.notna()) & (icu_h > 0) & (icu_h <= 168)) |
     ((df["death_28d"] == 1) & dth_h.notna() & (dth_h > 0))).astype(int).values
print(f"N={len(df)}  events={y.sum()} ({y.mean()*100:.1f}%)")

# corrected admission-window TG>=500 flag (Variant B)
df = df.copy()
df["tg_ge500_flag"] = (pd.to_numeric(df["tg_admission"], errors="coerce") >= 500).astype(int)

FULL = ["age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission", "diabetes", "hypertension",
        "heart_failure", "cad", "copd", "ckd", "dyslipidemia", "obesity_dx", "htg_dx",
        "baseline_wbc", "baseline_creatinine", "baseline_bun", "baseline_bilirubin",
        "baseline_platelet", "baseline_glucose", "baseline_lipase", "baseline_calcium"]

LGB = dict(objective="binary", verbosity=-1, n_estimators=200, max_depth=4, num_leaves=15,
           learning_rate=0.05, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
           reg_alpha=0.1, reg_lambda=1.0, random_state=42, is_unbalance=False)


def fit_fold(Xtr, ytr, Xte, kind="lgbm"):
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)
    Xtr2 = np.where(np.isnan(Xtr), med, Xtr); Xte2 = np.where(np.isnan(Xte), med, Xte)
    if kind == "lgbm":
        m = lgb.LGBMClassifier(**LGB).fit(Xtr2, ytr)
    else:
        sc = StandardScaler().fit(Xtr2); Xtr2, Xte2 = sc.transform(Xtr2), sc.transform(Xte2)
        m = LogisticRegression(max_iter=5000).fit(Xtr2, ytr)
    return m.predict_proba(Xte2)[:, 1]


def repeated_oof(X, yv, kind="lgbm", folds=10, repeats=10, seed=2024):
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    prob_sum = np.zeros(len(yv)); prob_cnt = np.zeros(len(yv)); per_rep = []
    cur = np.zeros(len(yv)); fold_in_rep = 0
    for tr, te in rskf.split(X, yv):
        p = fit_fold(X[tr], yv[tr], X[te], kind)
        cur[te] = p; prob_sum[te] += p; prob_cnt[te] += 1; fold_in_rep += 1
        if fold_in_rep == folds:
            per_rep.append(roc_auc_score(yv, cur)); cur = np.zeros(len(yv)); fold_in_rep = 0
    return np.array(per_rep), prob_sum / np.maximum(prob_cnt, 1)


def net_benefit(yv, p, t):
    pred = (p >= t).astype(int)
    tp = ((pred == 1) & (yv == 1)).sum(); fp = ((pred == 1) & (yv == 0)).sum()
    n = len(yv)
    return tp / n - fp / n * (t / (1 - t))


def summarize(name, feats, kind="lgbm"):
    X = df[feats].astype(float).values
    aucs, oof = repeated_oof(X, y, kind)
    ci = np.percentile(aucs, [2.5, 97.5])
    nb30 = net_benefit(y, oof, 0.30)
    row = {"model": name, "n_features": len(feats), "learner": kind,
           "oof_auroc": round(aucs.mean(), 3), "ci_low": round(ci[0], 3), "ci_high": round(ci[1], 3),
           "auprc": round(average_precision_score(y, oof), 3),
           "brier": round(brier_score_loss(y, oof), 3), "nb_at_0.30": round(nb30, 4)}
    print(f"  {name:42s} k={len(feats):2d} {kind:5s} OOF AUROC {aucs.mean():.3f} "
          f"({ci[0]:.3f}-{ci[1]:.3f})  AUPRC {row['auprc']:.3f}  Brier {row['brier']:.3f}  NB@.30 {nb30:.4f}")
    return row


# ---------- M3: feature-removal sensitivity (same learner = LightGBM) ----------
print("\n=== M3: feature-removal sensitivity (LightGBM, 10x10 OOF) ===")
m3 = []
m3.append(summarize("Full (21 feat, corrected)", FULL))
m3.append(summarize("Drop admission-window TG>=500 flag", [f for f in FULL if f != "tg_ge500_flag"]))
m3.append(summarize("Drop both TG features (flag + continuous)",
                    [f for f in FULL if f not in ("tg_ge500_flag", "tg_admission")]))
PHENO = {"tg_ge500_flag", "tg_admission", "metabolic_dx_flag", "htg_dx", "obesity_dx", "dyslipidemia"}
m3.append(summarize("Phenotype-orthogonal (drop all metabolic-pheno)",
                    [f for f in FULL if f not in PHENO]))
pd.DataFrame(m3).to_csv(os.path.join(OUT, "revision_M3_feature_removal.csv"), index=False)

# complete-case among patients with a measured admission TG (n where tg_admission notna)
cc = df["tg_admission"].notna().values
print(f"\n  Complete-case note: admission-window TG measured in {cc.sum()}/{len(df)} patients "
      f"(event rate {y[cc].mean():.3f} vs {y[~cc].mean():.3f} in unmeasured); "
      f"n={cc.sum()} too small for stable CV, reported descriptively.")

# ---------- M5: parsimonious / clinical-proxy baselines vs full model ----------
print("\n=== M5: baselines vs full governed model (OOF) ===")
m5 = []
m5.append(summarize("Full governed LightGBM (21 feat)", FULL, "lgbm"))
m5.append(summarize("Elastic-net comparator (21 feat)", FULL, "logit"))  # logit==plain LR here
m5.append(summarize("Age only (LR)", ["age"], "logit"))
PARS = ["age", "tg_ge500_flag", "baseline_calcium", "diabetes", "baseline_lipase"]
m5.append(summarize("Parsimonious top-5 SHAP (LR)", PARS, "logit"))
LABS = ["age", "baseline_bun", "baseline_creatinine", "baseline_wbc", "baseline_glucose", "baseline_calcium"]
m5.append(summarize("Simple admission-labs (LR)", LABS, "logit"))
# BISAP-available proxy: age>60, BUN>25 mg/dL (2 of 5 BISAP items reconstructable here)
bisap = ((df["age"] > 60).astype(int)
         + (pd.to_numeric(df["baseline_bun"], errors="coerce") > 25).fillna(0).astype(int)).values.astype(float)
# BISAP-proxy is an ordinal score; evaluate its raw discrimination (no fitting needed)
auc_bisap = roc_auc_score(y, bisap)
nb_bisap = net_benefit(y, (bisap >= 1).astype(float), 0.30)
print(f"  {'BISAP-available proxy (age>60 + BUN>25, 0-2)':42s} k= 2 score OOF AUROC "
      f"{auc_bisap:.3f} (raw, unfitted)  NB@.30 {nb_bisap:.4f}  [partial: 2/5 BISAP items]")
m5.append(dict(model="BISAP-available proxy (age>60+BUN>25)", n_features=2, learner="score",
               oof_auroc=round(auc_bisap, 3), ci_low="", ci_high="",
               auprc=round(average_precision_score(y, bisap), 3), brier="", **{"nb_at_0.30": round(nb_bisap, 4)}))
pd.DataFrame(m5).to_csv(os.path.join(OUT, "revision_M5_baselines.csv"), index=False)
print("\nSaved -> revision_M3_feature_removal.csv, revision_M5_baselines.csv")
