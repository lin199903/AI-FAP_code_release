"""
Emit OOF-based artifacts (corrected flag) for the manuscript figures/tables:
  - oof_predictions_corrected.csv      (per-patient mean OOF prob + outcome + probability summary)
  - governance_oof_calibration.csv     (No recal / Platt / Isotonic, cross-fitted on OOF)
  - governance_oof_abstention.csv      (abstention sweep on pooled OOF)
Repeated 10-fold x10 CV, per-fold median imputation (leakage-proof).
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))
icu_h, dth_h = df["icu_intime_hours"], df["death_offset_hours"]
y = (((icu_h.notna()) & (icu_h > 0) & (icu_h <= 168)) |
     ((df["death_28d"] == 1) & dth_h.notna() & (dth_h > 0))).astype(int).values
FEATS = ["age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission", "diabetes", "hypertension",
         "heart_failure", "cad", "copd", "ckd", "dyslipidemia", "obesity_dx", "htg_dx",
         "baseline_wbc", "baseline_creatinine", "baseline_bun", "baseline_bilirubin",
         "baseline_platelet", "baseline_glucose", "baseline_lipase", "baseline_calcium"]
X = df[FEATS].astype(float).values
LGB = dict(objective="binary", verbosity=-1, n_estimators=200, max_depth=4, num_leaves=15,
           learning_rate=0.05, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
           reg_alpha=0.1, reg_lambda=1.0, random_state=42, is_unbalance=False)

def ece(yt, p, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1])
        if m.sum(): e += m.sum() / len(p) * abs(yt[m].mean() - p[m].mean())
    return e

def cal(yt, p):
    eps = 1e-6; pp = np.clip(p, eps, 1 - eps); lg = np.log(pp / (1 - pp))
    lr = LogisticRegression(max_iter=1000).fit(lg.reshape(-1, 1), yt)
    return lr.intercept_[0], lr.coef_[0][0]

def fit_fold(Xtr, ytr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
    a = np.where(np.isnan(Xtr), med, Xtr); b = np.where(np.isnan(Xte), med, Xte)
    return lgb.LGBMClassifier(**LGB).fit(a, ytr).predict_proba(b)[:, 1]

# pooled OOF (mean over repeats) + per-repeat AUROC
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=2024)
psum = np.zeros(len(y)); pcnt = np.zeros(len(y)); per = []; cur = np.zeros(len(y)); k = 0
for tr, te in rskf.split(X, y):
    p = fit_fold(X[tr], y[tr], X[te]); cur[te] = p; psum[te] += p; pcnt[te] += 1; k += 1
    if k == 10: per.append(roc_auc_score(y, cur)); cur = np.zeros(len(y)); k = 0
oof = psum / np.maximum(pcnt, 1); per = np.array(per)
ci = np.percentile(per, [2.5, 97.5])

# These cut-points summarize predicted probabilities for plots and audit tables.
# They are not the 0.35 / 0.65 thresholds used in the final six-type bedside
# action map in 08_risk_typing_mapping.py.
SUMMARY_LOW_THRESHOLD = 0.20
SUMMARY_HIGH_THRESHOLD = 0.50

# per-patient OOF predictions + summary bands
strat = pd.cut(
    oof,
    [0, SUMMARY_LOW_THRESHOLD, SUMMARY_HIGH_THRESHOLD, 1.0],
    labels=["Low", "Intermediate", "High"],
    include_lowest=True,
)
pd.DataFrame({"hadm_id": df["hadm_id"], "prob_oof": oof, "composite_outcome": y,
             "probability_summary": strat, "risk_category": strat}).to_csv(os.path.join(OUT, "oof_predictions_corrected.csv"), index=False)

# cross-fitted recalibration scenarios
def crossfit(method):
    skf = StratifiedKFold(5, shuffle=True, random_state=7); out = np.zeros(len(y))
    for tr, te in skf.split(oof.reshape(-1, 1), y):
        if method == "iso":
            out[te] = IsotonicRegression(out_of_bounds="clip").fit(oof[tr], y[tr]).transform(oof[te])
        elif method == "platt":
            out[te] = LogisticRegression(max_iter=1000).fit(oof[tr].reshape(-1, 1), y[tr]).predict_proba(oof[te].reshape(-1, 1))[:, 1]
        else:
            out[te] = oof[te]
    return out
cal_rows = []
for name, m in [("No recalibration", "raw"), ("Platt scaling", "platt"), ("Isotonic regression", "iso")]:
    pp = crossfit(m); ci0, sl = cal(y, pp)
    cal_rows.append(dict(scenario=name, auroc=round(roc_auc_score(y, pp), 3), brier=round(brier_score_loss(y, pp), 3),
                         ece=round(ece(y, pp), 3), cal_intercept=round(ci0, 3), cal_slope=round(sl, 3)))
pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "governance_oof_calibration.csv"), index=False)

# abstention sweep on pooled OOF (withhold least-confident, |p-0.5| smallest)
conf = np.maximum(oof, 1 - oof); order = np.argsort(conf); ab_rows = []
for pct in [0, 5, 10, 15, 20, 30]:
    n = int(len(y) * pct / 100)
    keep = np.ones(len(y), bool)
    if n: keep[order[:n]] = False
    yt, pp = y[keep], oof[keep]
    ab_rows.append(dict(abstention_pct=pct, n_retained=int(keep.sum()), n_abstained=int(n),
                        auroc=round(roc_auc_score(yt, pp), 3), brier=round(brier_score_loss(yt, pp), 3),
                        ece=round(ece(yt, pp), 3),
                        outcome_rate_abstained=(round(y[~keep].mean(), 3) if n else None)))
pd.DataFrame(ab_rows).to_csv(os.path.join(OUT, "governance_oof_abstention.csv"), index=False)

print(f"OOF AUROC {per.mean():.3f} ({ci[0]:.3f}-{ci[1]:.3f})  AUPRC {average_precision_score(y, oof):.3f}  "
      f"Brier {brier_score_loss(y, oof):.3f}")
print("calibration scenarios:"); print(pd.DataFrame(cal_rows).to_string(index=False))
print("abstention:"); print(pd.DataFrame(ab_rows).to_string(index=False))
print(f"strata: " + ", ".join(f"{c}={int((strat==c).sum())}({100*y[strat==c].mean():.1f}%)" for c in ["Low","Intermediate","High"]))
