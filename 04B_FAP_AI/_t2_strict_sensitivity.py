# -*- coding: utf-8 -*-
"""
R2 防御：T0 入院模型的严格前瞻敏感性。
Primary  (L=0h) : 0-2h 入院面板特征 → 预测入院后任意时点(<=7d)的 early escalation/death
Strict   (L=2h) : 排除 0-2h 内已发生 ICU/死亡者，仅预测 2h 之后(<=7d)的事件
若 Strict 仍有合理 AUROC（即便略降），则 T0 信号非"识别已升级者"，而是真前瞻。
输入: outputs/canonical_mdap_cohort.csv
"""
import os, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))

BASELINE_FEATURES = [
    "age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission",
    "diabetes", "hypertension", "heart_failure", "cad", "copd", "ckd",
    "dyslipidemia", "obesity_dx", "htg_dx",
    "baseline_wbc", "baseline_creatinine", "baseline_bun",
    "baseline_bilirubin", "baseline_platelet", "baseline_glucose",
    "baseline_lipase", "baseline_calcium",
]
FEATS = [c for c in BASELINE_FEATURES if c in df.columns]
ICU_HORIZON_H = 168

# 时序 70/30（与 03 一致）
df["t0_dt"] = pd.to_datetime(df["t0"])
df = df.sort_values("t0_dt").reset_index(drop=True)
split = int(len(df) * 0.7)
train_ids = set(df.iloc[:split]["hadm_id"]); val_ids = set(df.iloc[split:]["hadm_id"])

def at_risk_outcome(d, Lh):
    icu_h, dth_h = d["icu_intime_hours"], d["death_offset_hours"]
    los_h = d["los_days"].astype(float) * 24.0
    if Lh <= 0:
        ar = pd.Series(True, index=d.index)
    else:
        ar = (icu_h.isna() | (icu_h > Lh)) & (dth_h.isna() | (dth_h > Lh)) & (los_h > Lh)
    icu_after = icu_h.notna() & (icu_h > Lh) & (icu_h <= ICU_HORIZON_H)
    death_after = (d["death_28d"] == 1) & dth_h.notna() & (dth_h > Lh)
    return ar, (icu_after | death_after).astype(int)

def logit_slope(y, p):
    eps = 1e-6; pl = np.log(np.clip(p, eps, 1-eps) / (1 - np.clip(p, eps, 1-eps)))
    lr = LogisticRegression(max_iter=1000).fit(pl.reshape(-1, 1), y)
    return lr.intercept_[0], lr.coef_[0][0]

def run(Lh, name):
    ar, y = at_risk_outcome(df, Lh)
    d = df[ar].copy(); d["_y"] = y[ar].values
    tr = d[d["hadm_id"].isin(train_ids)]; va = d[d["hadm_id"].isin(val_ids)]
    Xtr, Xva = tr[FEATS].copy(), va[FEATS].copy()
    for c in FEATS:
        m = Xtr[c].median(); m = 0 if pd.isna(m) else m
        Xtr[c] = Xtr[c].fillna(m); Xva[c] = Xva[c].fillna(m)
    ytr, yva = tr["_y"].values, va["_y"].values
    params = dict(objective="binary", n_estimators=200, max_depth=4, num_leaves=15,
                  learning_rate=0.05, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=42, is_unbalance=False, verbosity=-1)
    mdl = lgb.LGBMClassifier(**params).fit(Xtr, ytr)
    p = mdl.predict_proba(Xva)[:, 1]
    auroc = roc_auc_score(yva, p)
    rng = np.random.RandomState(42); boots = []
    for _ in range(1000):
        idx = rng.choice(len(yva), len(yva), replace=True)
        if yva[idx].sum() in (0, len(idx)): continue
        boots.append(roc_auc_score(yva[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    icpt, slp = logit_slope(yva, p)
    n_excl = (~ar).sum()
    print(f"\n[{name}]  L={Lh}h")
    print(f"  at-risk: {ar.sum()}/{len(df)}  (excluded {n_excl} who had event/left by {Lh}h)")
    print(f"  train {len(ytr)} (ev {int(ytr.sum())})  val {len(yva)} (ev {int(yva.sum())})")
    print(f"  AUROC {auroc:.3f} ({lo:.3f}-{hi:.3f})  Brier {brier_score_loss(yva,p):.3f}"
          f"  cal_slope {slp:.3f}")
    return auroc, lo, hi, int(yva.sum())

print("=" * 64); print("T0 入院模型 — 严格前瞻敏感性"); print("=" * 64)
run(0, "Primary (admission, predict any event <=7d)")
run(2, "Strict prospective (exclude 0-2h events, predict 2h..7d)")
