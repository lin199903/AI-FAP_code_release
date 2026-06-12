"""
Robustness pipeline for the T0 admission model (borrowed discipline: nested/repeated
CV + out-of-fold predictions + permutation test + per-fold imputation).

Purpose: replace the single temporal 70/30 split's high-variance estimate (41 val events)
with a stable OOF estimate, fit recalibration off OOF (not training predictions), and test
significance by label permutation. Run for BOTH the leaky any-time flag and the corrected
admission-window flag to give the definitive, resampling-stable effect of the leakage fix.

Key leakage-proof detail: training-set median imputation is fit INSIDE each fold and applied
to the held-out fold only. Nothing data-dependent crosses the fold boundary.

Outputs: outputs/robust_cv_oof_summary.csv  (one row per flag/model)
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))  # flag = corrected (Variant B)
icu_h, dth_h = df["icu_intime_hours"], df["death_offset_hours"]
y = (((icu_h.notna()) & (icu_h > 0) & (icu_h <= 168)) |
     ((df["death_28d"] == 1) & dth_h.notna() & (dth_h > 0))).astype(int).values

FEATS = ["age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission", "diabetes", "hypertension",
         "heart_failure", "cad", "copd", "ckd", "dyslipidemia", "obesity_dx", "htg_dx",
         "baseline_wbc", "baseline_creatinine", "baseline_bun", "baseline_bilirubin",
         "baseline_platelet", "baseline_glucose", "baseline_lipase", "baseline_calcium"]
corrected = (pd.to_numeric(df["tg_admission"], errors="coerce") >= 500).astype(int).values
leaky = df["tg_ge500_anytime_flag"].astype(int).values

LGB = dict(objective="binary", verbosity=-1, n_estimators=200, max_depth=4, num_leaves=15,
           learning_rate=0.05, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
           reg_alpha=0.1, reg_lambda=1.0, random_state=42, is_unbalance=False)

def ece(yt, p, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1])
        if m.sum(): e += m.sum() / len(p) * abs(yt[m].mean() - p[m].mean())
    return e

def cal_slope(yt, p):
    eps = 1e-6; pp = np.clip(p, eps, 1 - eps); lg = np.log(pp / (1 - pp))
    lr = LogisticRegression(max_iter=1000).fit(lg.reshape(-1, 1), yt)
    return lr.intercept_[0], lr.coef_[0][0]

def Xmat(fv):
    d = df.copy(); d["tg_ge500_flag"] = fv
    return d[FEATS].astype(float).values

def fit_fold(Xtr, ytr, Xte, kind="lgbm"):
    # per-fold median imputation fit on train only
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)
    Xtr2 = np.where(np.isnan(Xtr), med, Xtr); Xte2 = np.where(np.isnan(Xte), med, Xte)
    if kind == "lgbm":
        m = lgb.LGBMClassifier(**LGB).fit(Xtr2, ytr)
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr2); Xtr2, Xte2 = sc.transform(Xtr2), sc.transform(Xte2)
        m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000).fit(Xtr2, ytr)
    return m.predict_proba(Xte2)[:, 1]

def repeated_oof(X, yv, kind="lgbm", folds=10, repeats=10):
    """Return per-repeat OOF AUROC list and the mean OOF probability vector."""
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=2024)
    prob_sum = np.zeros(len(yv)); prob_cnt = np.zeros(len(yv)); per_rep = []
    cur = np.zeros(len(yv)); seen = np.zeros(len(yv), dtype=bool); fold_in_rep = 0
    for tr_idx, te_idx in rskf.split(X, yv):
        p = fit_fold(X[tr_idx], yv[tr_idx], X[te_idx], kind)
        cur[te_idx] = p; seen[te_idx] = True
        prob_sum[te_idx] += p; prob_cnt[te_idx] += 1
        fold_in_rep += 1
        if fold_in_rep == folds:  # one full repeat completed
            per_rep.append(roc_auc_score(yv, cur)); cur = np.zeros(len(yv)); seen[:] = False; fold_in_rep = 0
    return np.array(per_rep), prob_sum / np.maximum(prob_cnt, 1)

def crossfit_recal_ece(prob, yv, method="isotonic", folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=7); out = np.zeros(len(yv))
    for tr, te in skf.split(prob.reshape(-1, 1), yv):
        if method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip").fit(prob[tr], yv[tr]); out[te] = ir.transform(prob[te])
        else:
            lr = LogisticRegression(max_iter=1000).fit(prob[tr].reshape(-1, 1), yv[tr]); out[te] = lr.predict_proba(prob[te].reshape(-1, 1))[:, 1]
    return ece(yv, out)

rows = []
for tag, fv in [("leaky", leaky), ("corrected", corrected)]:
    X = Xmat(fv)
    aucs, oof = repeated_oof(X, y, "lgbm", folds=10, repeats=10)
    ci = np.percentile(aucs, [2.5, 97.5])
    ci0, sl = cal_slope(y, oof)
    raw_ece = ece(y, oof)
    iso_ece = crossfit_recal_ece(oof, y, "isotonic")
    platt_ece = crossfit_recal_ece(oof, y, "platt")
    # abstention on pooled OOF
    conf = np.maximum(oof, 1 - oof); order = np.argsort(conf); prev = y.mean()
    wr20 = y[order[:int(0.20 * len(y))]].mean(); wr30 = y[order[:int(0.30 * len(y))]].mean()
    print(f"\n=== {tag} flag ===")
    print(f"  OOF AUROC: {aucs.mean():.3f}  (repeat 2.5-97.5%: {ci[0]:.3f}-{ci[1]:.3f})  [10-fold x10 repeats]")
    print(f"  AUPRC {average_precision_score(y, oof):.3f}  Brier {brier_score_loss(y, oof):.3f}  "
          f"ECE(raw) {raw_ece:.3f}  cal_slope {sl:.3f}  intercept {ci0:.3f}")
    print(f"  cross-fitted recal ECE: Platt {platt_ece:.3f}  Isotonic {iso_ece:.3f}  "
          f"(raw {raw_ece:.3f}) -> {'helps' if min(platt_ece, iso_ece) < raw_ece else 'no improvement'}")
    print(f"  abstention withheld event rate (vs prevalence {prev:.3f}): @20%={wr20:.3f} @30%={wr30:.3f}  "
          f"-> {'paradox' if (wr20 > prev or wr30 > prev) else 'no paradox'}")
    rows.append(dict(flag=tag, oof_auroc=round(aucs.mean(), 3), ci_low=round(ci[0], 3), ci_high=round(ci[1], 3),
                     auprc=round(average_precision_score(y, oof), 3), brier=round(brier_score_loss(y, oof), 3),
                     ece_raw=round(raw_ece, 3), ece_platt=round(platt_ece, 3), ece_isotonic=round(iso_ece, 3),
                     cal_slope=round(sl, 3), cal_intercept=round(ci0, 3),
                     withheld20=round(wr20, 3), withheld30=round(wr30, 3), prevalence=round(prev, 3)))

# ---- permutation test (corrected flag) ----
print("\n=== Permutation test (corrected flag, 5-fold OOF AUROC, 1000 perms) ===")
Xc = Xmat(corrected)
obs_aucs, _ = repeated_oof(Xc, y, "lgbm", folds=5, repeats=4)
obs = obs_aucs.mean()
rng = np.random.RandomState(99); null = []
for i in range(1000):
    yp = rng.permutation(y)
    a, _ = repeated_oof(Xc, yp, "lgbm", folds=5, repeats=1)
    null.append(a[0])
null = np.array(null); pval = (1 + (null >= obs).sum()) / (1 + len(null))
print(f"  observed OOF AUROC {obs:.3f}  null mean {null.mean():.3f} (sd {null.std():.3f})  "
      f"null 95th pct {np.percentile(null, 95):.3f}  permutation p = {pval:.4f}")
for r in rows:
    r["perm_p_corrected"] = round(pval, 4) if r["flag"] == "corrected" else ""
pd.DataFrame(rows).to_csv(os.path.join(OUT, "robust_cv_oof_summary.csv"), index=False)
print("\nSaved -> outputs/robust_cv_oof_summary.csv")
