# -*- coding: utf-8 -*-
"""
AI-FAP 旗舰 — 地标 ML 动态再分层 + 校准 + SHAP + 风险分型
===============================================================================
按 AGENTS.md §6.2 / FLAGSHIP_ARCHITECTURE.md §6.6 执行动态地标机器学习。

设计（修正版，N=483 约束下克制建模）：
  三个地标点：T0（入院基线）、T0+24h、T0+48h
  模型：LightGBM（主）+ Elastic Net（校准基线）
  结局：复合结局 = ICU 7d OR 住院死亡（149/483 = 30.8%）
  划分：时序 70/30（最早 70% 训练，最新 30% 验证）
  SHAP：特征重要性 + 依赖图
  校准：Brier / ECE / calibration intercept & slope
  风险分型：基于预测概率的三类分型（Low / Intermediate / High）

输入：04B_FAP_AI/outputs/landmark_features_48h.csv
      04B_FAP_AI/outputs/landmark_features_24h.csv
      04B_FAP_AI/outputs/canonical_mdap_cohort.csv
输出：04B_FAP_AI/outputs/
  - landmark_ml_performance.csv     (三地标 × 两模型性能)
  - landmark_ml_shap_*.csv          (SHAP 特征重要性)
  - landmark_ml_predictions.csv     (每例预测概率 + 风险分型)
  - landmark_ml_calibration.png     (校准曲线)
  - landmark_ml_shap_bar.png        (SHAP 条形图)
  - landmark_ml_roc.png             (ROC 曲线)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, roc_curve, auc,
                              f1_score)
try:
    from sklearn.metrics import calibration_curve
except ImportError:
    from sklearn.calibration import calibration_curve
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. 数据加载与结局定义
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("AI-FAP Landmark ML Dynamic Restratification")
print("=" * 70)

df_cohort = pd.read_csv(os.path.join(OUTDIR, "canonical_mdap_cohort.csv"))
df_48h = pd.read_csv(os.path.join(OUTDIR, "landmark_features_48h.csv"))
df_24h = pd.read_csv(os.path.join(OUTDIR, "landmark_features_24h.csv"))

print(f"\nCohort: {len(df_cohort)} admissions")

# Composite outcome: ICU within 7d OR hospital mortality
df_cohort["composite_outcome"] = (
    (df_cohort["icu_7d"] == 1) | (df_cohort["hospital_expire_flag"] == 1)
).astype(int)
n_events = df_cohort["composite_outcome"].sum()
print(f"Composite outcome (ICU 7d OR hosp death): {n_events}/{len(df_cohort)} "
      f"({100*n_events/len(df_cohort):.1f}%)")

# ── 防泄漏：地标-相对 at-risk 与结局 ─────────────────────────
# 在地标 L（小时）处，只纳入仍 at-risk（未进 ICU、未死亡、仍在院）的患者，
# 且只预测 L 之后发生的事件。否则 T48 会用 0-48h 数据"预测"窗口内已发生的事件。
LANDMARK_HOURS = {"T0": 0, "T24": 24, "T48": 48}
ICU_HORIZON_H = 168  # ICU 结局地平线 = 入院后 7 天

def landmark_at_risk_outcome(df, Lh):
    """返回 (at_risk: bool Series, y: int Series) — 地标-相对。"""
    icu_h = df["icu_intime_hours"]
    dth_h = df["death_offset_hours"]
    los_h = df["los_days"].astype(float) * 24.0
    if Lh <= 0:
        at_risk = pd.Series(True, index=df.index)
    else:
        at_risk = (
            (icu_h.isna() | (icu_h > Lh)) &      # L 前未进 ICU
            (dth_h.isna() | (dth_h > Lh)) &      # L 前未死亡
            (los_h > Lh)                         # L 时仍在院
        )
    icu_after = icu_h.notna() & (icu_h > Lh) & (icu_h <= ICU_HORIZON_H)
    death_after = (df["death_28d"] == 1) & dth_h.notna() & (dth_h > Lh)
    y = (icu_after | death_after).astype(int)
    return at_risk, y

OUTCOME_COLS = ["hadm_id", "icu_intime_hours", "death_offset_hours",
                "death_28d", "los_days", "composite_outcome", "t0_dt"]

# ═══════════════════════════════════════════════════════════
# 2. 时序划分
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 2: Temporal Train/Validation Split")
print("=" * 70)

df_cohort["t0_dt"] = pd.to_datetime(df_cohort["t0"])
df_cohort = df_cohort.sort_values("t0_dt").reset_index(drop=True)

split_idx = int(len(df_cohort) * 0.7)
train_hadm = set(df_cohort.iloc[:split_idx]["hadm_id"].values)
val_hadm = set(df_cohort.iloc[split_idx:]["hadm_id"].values)

print(f"  Train: {len(train_hadm)} (t0 <= {df_cohort.iloc[split_idx-1]['t0']})")
print(f"  Val:   {len(val_hadm)} (t0 > {df_cohort.iloc[split_idx]['t0']})")
print(f"  Train events: {df_cohort.iloc[:split_idx]['composite_outcome'].sum()}")
print(f"  Val events:   {df_cohort.iloc[split_idx:]['composite_outcome'].sum()}")

# ═══════════════════════════════════════════════════════════
# 3. 地标特征集定义
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 3: Landmark Feature Set Definition")
print("=" * 70)

BASELINE_FEATURES = [
    "age", "tg_ge500_flag", "metabolic_dx_flag", "tg_admission",
    "diabetes", "hypertension", "heart_failure", "cad", "copd", "ckd",
    "dyslipidemia", "obesity_dx", "htg_dx",
    "baseline_wbc", "baseline_creatinine", "baseline_bun",
    "baseline_bilirubin", "baseline_platelet", "baseline_glucose",
    "baseline_lipase", "baseline_calcium",
]

TRAJ_MARKERS_CORE = ["wbc", "creatinine", "bun", "platelet", "glucose"]
TRAJ_MARKERS_OPT = ["lactate", "bilirubin"]
TRAJ_MARKERS_ALL = TRAJ_MARKERS_CORE + TRAJ_MARKERS_OPT

def build_landmark_features(df_feat, landmark, include_gbtm=False):
    """Build feature matrix for a given landmark time point."""
    features = list(BASELINE_FEATURES)

    if landmark == "T0":
        pass
    elif landmark == "T24":
        for m in TRAJ_MARKERS_ALL:
            for w in ["w0_6", "w6_24"]:
                col = f"{m}_{w}"
                if col in df_feat.columns:
                    features.append(col)
            if f"{m}_slope" in df_feat.columns:
                features.append(f"{m}_slope")
            if f"{m}_max" in df_feat.columns:
                features.append(f"{m}_max")
    elif landmark == "T48":
        for m in TRAJ_MARKERS_ALL:
            for w in ["w0_6", "w6_24", "w24_48"]:
                col = f"{m}_{w}"
                if col in df_feat.columns:
                    features.append(col)
            if f"{m}_slope" in df_feat.columns:
                features.append(f"{m}_slope")
            if f"{m}_max" in df_feat.columns:
                features.append(f"{m}_max")
            if f"{m}_delta_pct" in df_feat.columns:
                features.append(f"{m}_delta_pct")
        if include_gbtm and "gbtm_class" in df_feat.columns:
            features.append("gbtm_class")

    features = [f for f in features if f in df_feat.columns]
    return features

landmarks = {
    "T0": {"df": df_cohort, "include_gbtm": False},   # 统一真相源：T0 基线特征直取 canonical（+2h 窗），消除派生分叉
    "T24": {"df": df_24h, "include_gbtm": False},
    "T48": {"df": df_48h, "include_gbtm": False},  # 防泄漏：gbtm_class 用全 0-48h 拟合，不得作特征
}

for lm_name, lm_cfg in landmarks.items():
    feats = build_landmark_features(lm_cfg["df"], lm_name, lm_cfg["include_gbtm"])
    df_lm = lm_cfg["df"][feats].copy()
    coverage = df_lm.notna().mean()
    high_cov = (coverage >= 0.5).sum()
    print(f"  {lm_name}: {len(feats)} features, {high_cov} with >=50% coverage")

# ═══════════════════════════════════════════════════════════
# 4. 模型训练与评估
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 4: Model Training & Evaluation")
print("=" * 70)

def prepare_features(df_feat, feature_list, train_ids, val_ids):
    """Prepare feature matrices with missing value handling."""
    df_feat = df_feat.copy()

    # Gender encoding
    if "gender" in feature_list:
        le = LabelEncoder()
        df_feat["gender"] = le.fit_transform(df_feat["gender"].fillna("M"))

    # GBTM class: fill NaN with -1 (not eligible)
    if "gbtm_class" in feature_list:
        df_feat["gbtm_class"] = df_feat["gbtm_class"].fillna(-1).astype(int)

    # Split
    train_mask = df_feat["hadm_id"].isin(train_ids)
    val_mask = df_feat["hadm_id"].isin(val_ids)

    X_train = df_feat.loc[train_mask, feature_list].copy()
    X_val = df_feat.loc[val_mask, feature_list].copy()

    # Fill remaining NaN with median from training set
    for col in feature_list:
        if col == "hadm_id":
            continue
        med = X_train[col].median()
        if pd.isna(med):
            med = 0
        X_train[col] = X_train[col].fillna(med)
        X_val[col] = X_val[col].fillna(med)

    # Drop hadm_id if present
    if "hadm_id" in X_train.columns:
        X_train = X_train.drop(columns=["hadm_id"])
        X_val = X_val.drop(columns=["hadm_id"])

    return X_train, X_val, train_mask, val_mask


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_prob) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def calibration_metrics(y_true, y_prob):
    """标准 calibration intercept/slope：回归在 logit(p) 上（非原始概率）。"""
    from sklearn.linear_model import LogisticRegression as LR
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    lr = LR(fit_intercept=True, max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y_true)
    return lr.intercept_[0], lr.coef_[0][0]


results = []
all_predictions = {}
shap_data = {}

for lm_name, lm_cfg in landmarks.items():
    print(f"\n--- Landmark: {lm_name} ---")
    df_feat = lm_cfg["df"]
    feature_list = build_landmark_features(df_feat, lm_name, lm_cfg["include_gbtm"])

    # Remove hadm_id from feature list for modeling
    feat_cols = [f for f in feature_list if f != "hadm_id"]

    # Add outcome + landmark timing（先丢掉重叠列，确保取 canonical 的新列）
    _overlap = [c for c in OUTCOME_COLS if c != "hadm_id" and c in df_feat.columns]
    df_feat = df_feat.drop(columns=_overlap).merge(
        df_cohort[OUTCOME_COLS], on="hadm_id", how="left")

    # 地标-相对 at-risk 与结局（防泄漏）
    Lh = LANDMARK_HOURS[lm_name]
    at_risk, y_lm = landmark_at_risk_outcome(df_feat, Lh)
    df_feat = df_feat.assign(_y=y_lm.values)
    n_before = len(df_feat)
    df_feat = df_feat[at_risk.values].reset_index(drop=True)
    print(f"  At-risk @ {lm_name}(+{Lh}h): {len(df_feat)}/{n_before} "
          f"(excluded {n_before - len(df_feat)} who already had event / left before L)")

    X_train, X_val, train_mask, val_mask = prepare_features(
        df_feat, feature_list, train_hadm, val_hadm
    )

    y_train = df_feat.loc[train_mask, "_y"].values
    y_val = df_feat.loc[val_mask, "_y"].values

    print(f"  Train: {len(y_train)} (events={int(y_train.sum())}, {100*y_train.mean():.1f}%)")
    print(f"  Val:   {len(y_val)} (events={int(y_val.sum())}, {100*y_val.mean():.1f}%)")
    print(f"  Features: {X_train.shape[1]}")

    if len(set(y_train)) < 2 or len(set(y_val)) < 2:
        print(f"  [SKIP] {lm_name}: 训练/验证集结局单一类，at-risk 后事件不足，无法建模。")
        results.append({"landmark": lm_name, "model": "LightGBM",
                        "n_features": X_train.shape[1], "auroc": None,
                        "n_train": len(y_train), "n_val": len(y_val),
                        "n_events_train": int(y_train.sum()),
                        "n_events_val": int(y_val.sum())})
        continue

    # ── LightGBM ──────────────────────────────────────────
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_estimators": 200,
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "is_unbalance": False,   # 防校准失真：不做类加权，概率校准到真实基率；不平衡交由重校准层处理
    }

    # 防 val 泄漏：不在验证集上 early stopping（否则 n_estimators 被 val 调参 → 乐观）
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train)

    prob_lgb_train = model_lgb.predict_proba(X_train)[:, 1]
    prob_lgb_val = model_lgb.predict_proba(X_val)[:, 1]

    auroc_lgb = roc_auc_score(y_val, prob_lgb_val)
    auprc_lgb = average_precision_score(y_val, prob_lgb_val)
    brier_lgb = brier_score_loss(y_val, prob_lgb_val)
    ece_lgb = compute_ece(y_val, prob_lgb_val)
    cal_int_lgb, cal_slope_lgb = calibration_metrics(y_val, prob_lgb_val)

    print(f"  LightGBM: AUROC={auroc_lgb:.3f}, AUPRC={auprc_lgb:.3f}, "
          f"Brier={brier_lgb:.3f}, ECE={ece_lgb:.3f}")
    print(f"    Calibration: intercept={cal_int_lgb:.3f}, slope={cal_slope_lgb:.3f}")

    results.append({
        "landmark": lm_name, "model": "LightGBM",
        "n_features": X_train.shape[1],
        "auroc": round(auroc_lgb, 3),
        "auprc": round(auprc_lgb, 3),
        "brier": round(brier_lgb, 3),
        "ece": round(ece_lgb, 3),
        "cal_intercept": round(cal_int_lgb, 3),
        "cal_slope": round(cal_slope_lgb, 3),
        "n_train": len(y_train), "n_val": len(y_val),
        "n_events_train": int(y_train.sum()),
        "n_events_val": int(y_val.sum()),
    })

    # ── Elastic Net (logistic) ────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    model_en = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5,
        C=1.0, max_iter=5000, random_state=42, class_weight=None
    )
    model_en.fit(X_train_sc, y_train)

    prob_en_val = model_en.predict_proba(X_val_sc)[:, 1]

    auroc_en = roc_auc_score(y_val, prob_en_val)
    auprc_en = average_precision_score(y_val, prob_en_val)
    brier_en = brier_score_loss(y_val, prob_en_val)
    ece_en = compute_ece(y_val, prob_en_val)
    cal_int_en, cal_slope_en = calibration_metrics(y_val, prob_en_val)

    print(f"  ElasticNet: AUROC={auroc_en:.3f}, AUPRC={auprc_en:.3f}, "
          f"Brier={brier_en:.3f}, ECE={ece_en:.3f}")

    results.append({
        "landmark": lm_name, "model": "ElasticNet",
        "n_features": X_train.shape[1],
        "auroc": round(auroc_en, 3),
        "auprc": round(auprc_en, 3),
        "brier": round(brier_en, 3),
        "ece": round(ece_en, 3),
        "cal_intercept": round(cal_int_en, 3),
        "cal_slope": round(cal_slope_en, 3),
        "n_train": len(y_train), "n_val": len(y_val),
        "n_events_train": int(y_train.sum()),
        "n_events_val": int(y_val.sum()),
    })

    # ── SHAP (LightGBM) ──────────────────────────────────
    try:
        explainer = shap.TreeExplainer(model_lgb)
        shap_values = explainer.shap_values(X_val)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": X_train.columns,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        shap_data[lm_name] = {
            "shap_values": shap_values,
            "features": X_train.columns,
            "X_val": X_val,
            "shap_df": shap_df,
        }

        shap_path = os.path.join(OUTDIR, f"landmark_ml_shap_{lm_name}.csv")
        shap_df.to_csv(shap_path, index=False)
        print(f"  SHAP top-5: {', '.join(shap_df.head(5)['feature'].tolist())}")
    except Exception as e:
        print(f"  SHAP failed: {e}")

    # ── Store predictions ────────────────────────────────
    val_hadm_arr = df_feat.loc[val_mask, "hadm_id"].values
    pred_df = pd.DataFrame({
        "hadm_id": val_hadm_arr,
        f"prob_lgb_{lm_name}": prob_lgb_val,
        f"prob_en_{lm_name}": prob_en_val,
        "composite_outcome": y_val,
    })
    all_predictions[lm_name] = pred_df

# ═══════════════════════════════════════════════════════════
# 5. 风险分型
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 5: Probability Summary Bands (3-class)")
print("=" * 70)

# 可用地标（at-risk 过滤后可能有地标被跳过）
AVAIL = [lm for lm in ["T0", "T24", "T48"] if lm in all_predictions]
if not AVAIL:
    print("  [FATAL] 所有地标 at-risk 后均无足够事件，无法风险分型。")
    sys.exit(1)
# 主命题已转向入院基线分型：T0 是统计可支撑的主模型（48h 事件前置，T24/T48 at-risk 事件不足）
PRIMARY_LM = "T0" if "T0" in AVAIL else AVAIL[-1]
print(f"  Primary landmark for risk typing: {PRIMARY_LM} (入院基线分型；T24/T48 仅作动态增益的阴性证据)")

pred_t48 = all_predictions[PRIMARY_LM].copy()
prob_primary = pred_t48[f"prob_lgb_{PRIMARY_LM}"].values

# These cut-points summarize probability strata (Low / Intermediate / High)
# for plots and transportability summaries only. They are distinct from the
# conservative 0.35 / 0.65 gray-zone thresholds used later in the six-type
# bedside action map in 08_risk_typing_mapping.py.
SUMMARY_LOW_THRESHOLD = 0.20
SUMMARY_HIGH_THRESHOLD = 0.50

pred_t48["probability_summary"] = pd.cut(
    prob_primary,
    bins=[0, SUMMARY_LOW_THRESHOLD, SUMMARY_HIGH_THRESHOLD, 1.0],
    labels=["Low", "Intermediate", "High"],
    include_lowest=True,
)
pred_t48["risk_category"] = pred_t48["probability_summary"]

print(f"\n  Probability summary distribution ({PRIMARY_LM}, validation set):")
for cat in ["Low", "Intermediate", "High"]:
    mask = pred_t48["probability_summary"] == cat
    n = mask.sum()
    events = pred_t48.loc[mask, "composite_outcome"].sum()
    rate = 100 * events / n if n > 0 else 0
    print(f"    {cat}: n={n}, events={events} ({rate:.1f}%)")

# Merge with GBTM class
gbtm_assign = pd.read_csv(os.path.join(OUTDIR, "gbtm_trajectory_assignments.csv"))
pred_t48 = pred_t48.merge(gbtm_assign[["hadm_id", "gbtm_class", "gbtm_class_name"]],
                           on="hadm_id", how="left")

# Cross-tabulation: risk category × GBTM class
print(f"\n  Probability summary × GBTM class:")
ct = pd.crosstab(pred_t48["probability_summary"], pred_t48["gbtm_class_name"], margins=True)
print(ct.to_string())

pred_path = os.path.join(OUTDIR, "landmark_ml_predictions.csv")
pred_t48.to_csv(pred_path, index=False)
print(f"\n  Predictions saved to {pred_path}")

# ═══════════════════════════════════════════════════════════
# 6. 性能汇总表
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 6: Performance Summary")
print("=" * 70)

df_results = pd.DataFrame(results)
perf_path = os.path.join(OUTDIR, "landmark_ml_performance.csv")
df_results.to_csv(perf_path, index=False)
print(df_results.to_string(index=False))
print(f"\n  Saved to {perf_path}")

# ═══════════════════════════════════════════════════════════
# 7. 可视化
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 7: Visualization")
print("=" * 70)

COLORS_LM = {"T0": "#2196F3", "T24": "#FF9800", "T48": "#F44336"}
COLORS_MODEL = {"LightGBM": "#2196F3", "ElasticNet": "#FF9800"}

# ── 7a. ROC curves ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("AI-FAP: Landmark ML Performance", fontsize=14, fontweight="bold")

for model_name, ax in zip(["LightGBM", "ElasticNet"], axes):
    for lm_name in AVAIL:
        pred = all_predictions[lm_name]
        prob_col = f"prob_{model_name.lower().replace('elasticnet','en')}_{lm_name}"
        if model_name == "ElasticNet":
            prob_col = f"prob_en_{lm_name}"
        else:
            prob_col = f"prob_lgb_{lm_name}"

        y_true = pred["composite_outcome"].values
        y_prob = pred[prob_col].values

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)

        ax.plot(fpr, tpr, color=COLORS_LM[lm_name], linewidth=2,
                label=f"{lm_name} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 - Specificity", fontsize=11)
    ax.set_ylabel("Sensitivity", fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(OUTDIR, "landmark_ml_roc.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  ROC curves saved to {fig_path}")

# ── 7b. Calibration curves ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("AI-FAP: Calibration Curves (LightGBM)", fontsize=14, fontweight="bold")

for idx, lm_name in enumerate(AVAIL):
    ax = axes[idx]
    pred = all_predictions[lm_name]
    y_true = pred["composite_outcome"].values
    y_prob = pred[f"prob_lgb_{lm_name}"].values

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")

    ax.plot(prob_pred, prob_true, "o-", color=COLORS_LM[lm_name], linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Observed frequency", fontsize=11)
    ax.set_title(f"{lm_name} (Brier={brier_score_loss(y_true, y_prob):.3f})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(OUTDIR, "landmark_ml_calibration.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Calibration curves saved to {fig_path}")

# ── 7c. SHAP bar plots ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("AI-FAP: SHAP Feature Importance (LightGBM)", fontsize=14, fontweight="bold")

for idx, lm_name in enumerate(AVAIL):
    ax = axes[idx]
    if lm_name not in shap_data:
        ax.text(0.5, 0.5, "SHAP unavailable", ha="center", va="center")
        continue

    shap_df = shap_data[lm_name]["shap_df"]
    top_n = min(15, len(shap_df))
    top_features = shap_df.head(top_n)

    bars = ax.barh(range(top_n), top_features["mean_abs_shap"].values[::-1],
                   color=COLORS_LM[lm_name], alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features["feature"].values[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(lm_name, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(OUTDIR, "landmark_ml_shap_bar.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  SHAP bar plots saved to {fig_path}")

# ── 7d. Probability summary × outcome ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("AI-FAP: Risk Category Performance", fontsize=14, fontweight="bold")

# Bar chart: outcome rate by risk category
ax = axes[0]
cats = ["Low", "Intermediate", "High"]
rates = []
for cat in cats:
    mask = pred_t48["probability_summary"] == cat
    n = mask.sum()
    events = pred_t48.loc[mask, "composite_outcome"].sum()
    rates.append(100 * events / n if n > 0 else 0)

bar_colors = ["#4CAF50", "#FF9800", "#F44336"]
bars = ax.bar(cats, rates, color=bar_colors, edgecolor="black", linewidth=0.5)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Composite outcome rate (%)", fontsize=11)
ax.set_title("Outcome by Risk Category (T48)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# AUROC across landmarks
ax = axes[1]
for model_name in ["LightGBM", "ElasticNet"]:
    model_results = df_results[df_results["model"] == model_name]
    lms = model_results["landmark"].values
    aurocs = model_results["auroc"].values
    ax.plot(lms, aurocs, "o-", label=model_name, linewidth=2, markersize=8)

ax.set_xlabel("Landmark", fontsize=11)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_title("AUROC Across Landmarks", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.0)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(OUTDIR, "landmark_ml_risk_categories.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Probability summary plot saved to {fig_path}")

# ═══════════════════════════════════════════════════════════
# 8. Bootstrap AUROC 置信区间
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 8: Bootstrap AUROC 95% CI")
print("=" * 70)

N_BOOT = 1000
np.random.seed(42)

for lm_name in AVAIL:
    pred = all_predictions[lm_name]
    y_true = pred["composite_outcome"].values
    y_prob = pred[f"prob_lgb_{lm_name}"].values

    boot_aurocs = []
    for _ in range(N_BOOT):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if y_true[idx].sum() == 0 or (1 - y_true[idx]).sum() == 0:
            continue
        boot_aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    ci_low = np.percentile(boot_aurocs, 2.5)
    ci_high = np.percentile(boot_aurocs, 97.5)
    point_est = roc_auc_score(y_true, y_prob)
    print(f"  {lm_name} LightGBM AUROC: {point_est:.3f} ({ci_low:.3f}-{ci_high:.3f})")

# ═══════════════════════════════════════════════════════════
# 9. 汇总
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Landmark ML Dynamic Restratification Complete
──────────────────────────────────────────────
Cohort:             {len(df_cohort)} MDAP admissions
Composite outcome:  {n_events} ({100*n_events/len(df_cohort):.1f}%)
Train/Val split:    {len(train_hadm)}/{len(val_hadm)} (temporal 70/30)

Output files:
  landmark_ml_performance.csv     - performance metrics
  landmark_ml_predictions.csv     - predictions + probability summary bands
  landmark_ml_shap_T0/T24/T48.csv - SHAP feature importance
  landmark_ml_roc.png             - ROC curves
  landmark_ml_calibration.png     - calibration curves
  landmark_ml_shap_bar.png        - SHAP bar plots
  landmark_ml_risk_categories.png - risk category outcomes
""")

print("Next: AI governance layer (calibration, abstention, shortcut probes)")
