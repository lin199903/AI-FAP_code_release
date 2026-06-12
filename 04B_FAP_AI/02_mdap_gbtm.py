# -*- coding: utf-8 -*-
"""
AI-FAP 旗舰 — GBTM 轨迹建模 + 轨迹表型描述 + 地标特征工程
===============================================================================
按 AGENTS.md §6.2 / FLAGSHIP_ARCHITECTURE.md §6.5 执行 48h 炎症-器官功能轨迹建模。

=== AUDIT TRAIL / GAP NOTE (added by Grok 2026-06 per user "you execute first") ===
This module implements the GBTM phenotyping required by §6.2.
However, per detailed gap analysis in manuscript/JBI/FLAG SHIP_GAP_vs_AGENTS_S6.md:
- Full §6 requires integration with 6 risk types (Routine/Metabolic-vulnerable/etc.) in governance layer,
  Tier 3 action mapping (monitoring intensity only), shortcut probe for measurement bias,
  and explicit de-escalation audit after 48h.
- Current JBI MS is strong on admission + leakage but does not yet deliver the "48h trajectory mother body
  turned into governable monitoring-priority typing".
- TODO for next implementation round (after this verification): extend outputs to feed 04_ai_governance.py,
  produce the 6-type table + main flow figure, update JBI .tex positioning.
See VERIFICATION_REPORT_BY_GROK.md and the 1-page prompt for context.
This edit is part of "you execute first" — concrete placeholder added instead of only prompt.
===============================================================================

主轴（已锁定，G1=226 否决 TG 轨迹）：
  炎症/器官功能轨迹（WBC、Cr、BUN、血小板、血糖）
  时间窗：6-24h / 24-48h（w0_6 覆盖率仅 25-39%，不纳入主分析）
  TG 仅作入院基线表型变量

方法：
  GBTM（Group-Based Trajectory Model, Nagin 2005）
  Python 实现：sklearn.mixture.GaussianMixture + 自写 EM 迭代
  测试 2-5 类解，选 BIC 最优 + 临床可解释

输入：04B_FAP_AI/outputs/canonical_mdap_cohort.csv
输出：04B_FAP_AI/outputs/
  - gbtm_trajectory_assignments.csv   (每 hadm_id 的轨迹类分配 + 后验概率)
  - gbtm_class_profiles.csv           (每类的时间点均值 + 95%CI)
  - gbtm_model_comparison.csv         (2-5 类解的 BIC/AIC/entropy/最小类占比)
  - gbtm_trajectory_plot.png          (轨迹图)
  - gbtm_class_outcome_table.csv      (轨迹类 × 结局交叉表)
  - landmark_features_24h.csv         (T0+24h 地标特征集)
  - landmark_features_48h.csv         (T0+48h 地标特征集)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

COHORT_PATH = os.path.join(OUTDIR, "canonical_mdap_cohort.csv")

# ═══════════════════════════════════════════════════════════
# 1. 数据加载与预处理
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("AI-FAP GBTM Trajectory Modeling")
print("=" * 70)

df = pd.read_csv(COHORT_PATH)
print(f"\nLoaded: {df.shape[0]} admissions, {df.shape[1]} columns")

TRAJ_MARKERS = ["wbc", "creatinine", "bun", "platelet", "glucose"]
OPTIONAL_MARKERS = ["lactate", "bilirubin"]
ALL_TRAJ_MARKERS = TRAJ_MARKERS + OPTIONAL_MARKERS

WINDOWS = ["w6_24", "w24_48"]
WINDOW_LABELS = ["6-24h", "24-48h"]
WINDOW_MIDS = [15.0, 36.0]

print(f"\nTrajectory markers (core): {TRAJ_MARKERS}")
print(f"Trajectory markers (optional): {OPTIONAL_MARKERS}")
print(f"Time windows: {WINDOWS}")

# ── 1a. 构建轨迹矩阵 ─────────────────────────────────────
traj_cols = []
for m in TRAJ_MARKERS:
    for w in WINDOWS:
        traj_cols.append(f"{m}_{w}")

print(f"\nCore trajectory columns: {len(traj_cols)}")
for c in traj_cols:
    n = df[c].notna().sum()
    print(f"  {c}: {n}/{len(df)} ({100*n/len(df):.1f}%)")

# ── 1b. 缺失处理策略 ──────────────────────────────────────
# GBTM 要求完整轨迹。策略：
#   主分析：要求 w6_24 + w24_48 两个窗口均有值（至少 core markers）
#   敏感性：单窗口插补（用基线值或前窗值）

MIN_CORE_COMPLETE = 4  # 至少 4/5 core markers 在两个窗口均有值

def check_trajectory_completeness(row):
    complete = 0
    for m in TRAJ_MARKERS:
        has_both = all(pd.notna(row[f"{m}_{w}"]) for w in WINDOWS)
        if has_both:
            complete += 1
    return complete

df["traj_complete_count"] = df.apply(check_trajectory_completeness, axis=1)
df["traj_eligible"] = df["traj_complete_count"] >= MIN_CORE_COMPLETE

n_eligible = df["traj_eligible"].sum()
print(f"\nTrajectory eligibility (>= {MIN_CORE_COMPLETE}/5 core markers complete):")
print(f"  Eligible: {n_eligible}/{len(df)} ({100*n_eligible/len(df):.1f}%)")
print(f"  Excluded: {(~df['traj_eligible']).sum()}")

df_traj = df[df["traj_eligible"]].copy().reset_index(drop=True)
print(f"\nGBTM analysis cohort: {len(df_traj)} admissions")

# ── 1c. 缺失 marker 插补（仅对 eligible 队列中零星缺失） ──
# 策略：用同 marker 的可用窗口均值 → 基线值 → 队列中位数
for m in TRAJ_MARKERS:
    for w in WINDOWS:
        col = f"{m}_{w}"
        mask = df_traj[col].isna()
        if mask.sum() == 0:
            continue
        baseline_col = f"baseline_{m}"
        for idx in df_traj[mask].index:
            other_w = [ww for ww in WINDOWS if ww != w][0]
            other_val = df_traj.loc[idx, f"{m}_{other_w}"]
            if pd.notna(other_val):
                df_traj.loc[idx, col] = other_val
            elif baseline_col in df_traj.columns and pd.notna(df_traj.loc[idx, baseline_col]):
                df_traj.loc[idx, col] = df_traj.loc[idx, baseline_col]
            else:
                df_traj.loc[idx, col] = df_traj[col].median()
        print(f"  Imputed {col}: {mask.sum()} values")

# ═══════════════════════════════════════════════════════════
# 2. GBTM 建模 — 单变量轨迹（每 marker 独立）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 2: Univariate GBTM (per marker)")
print("=" * 70)

class GBTMUnivariate:
    """Group-Based Trajectory Model for a single continuous marker.
    
    Nagin (2005) framework using Gaussian Mixture on time-series features.
    Input: N × T matrix (N subjects, T time points).
    Each class has a polynomial trajectory (intercept + slope + ...).
    """

    def __init__(self, n_classes, polynomial_order=1, n_init=20, max_iter=500, random_state=42):
        self.n_classes = n_classes
        self.poly_order = polynomial_order
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.fitted = False

    def fit(self, Y, time_points=None):
        """Y: (N, T) array of observed values. time_points: (T,) array."""
        self.N, self.T = Y.shape
        if time_points is None:
            time_points = np.arange(self.T, dtype=float)
        self.time_points = time_points

        # Design matrix: polynomial in time
        self.X = np.column_stack(
            [time_points ** k for k in range(self.poly_order + 1)]
        )  # (T, poly_order+1)

        # Initialize with GMM on raw data
        Y_flat = Y.flatten().reshape(-1, 1)
        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y_flat)

        gmm = GaussianMixture(
            n_components=self.n_classes,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            covariance_type="full",
        )
        gmm.fit(Y_scaled)

        # EM iterations for trajectory model
        self._fit_em(Y)

        self.fitted = True
        return self

    def _fit_em(self, Y):
        """EM algorithm for GBTM."""
        N, T = self.N, self.T
        K = self.n_classes
        X = self.X  # (T, P)

        # Initialize class assignments from GMM
        Y_flat = Y.flatten().reshape(-1, 1)
        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y_flat)
        gmm = GaussianMixture(n_components=K, n_init=self.n_init,
                               max_iter=self.max_iter, random_state=self.random_state)
        gmm.fit(Y_scaled)
        init_labels = gmm.predict(Y_scaled).reshape(N, T)

        # Majority vote per subject
        self.pi = np.zeros(K)
        self.beta = np.zeros((K, self.poly_order + 1))
        self.sigma2 = np.zeros(K)

        for k in range(K):
            mask = (init_labels == k)
            self.pi[k] = mask.any(axis=1).sum() / N
            if self.pi[k] < 0.01:
                self.pi[k] = 0.01
            y_k = Y[mask]
            if len(y_k) > self.poly_order + 1:
                t_k = np.tile(self.time_points, N)[mask.flatten()]
                self.beta[k] = np.linalg.lstsq(
                    np.column_stack([t_k ** p for p in range(self.poly_order + 1)]),
                    y_k, rcond=None
                )[0]
            else:
                self.beta[k, 0] = Y.mean()
            self.sigma2[k] = max(np.var(Y[mask]) if mask.sum() > 1 else np.var(Y), 0.01)

        self.pi /= self.pi.sum()

        # EM iterations
        for iteration in range(self.max_iter):
            # E-step: compute posterior probabilities
            log_prob = np.zeros((N, K))
            for k in range(K):
                mu_k = X @ self.beta[k]  # (T,)
                resid = Y - mu_k[np.newaxis, :]
                log_pdf = -0.5 * np.log(self.sigma2[k]) - 0.5 * resid ** 2 / self.sigma2[k]
                log_prob[:, k] = np.sum(log_pdf, axis=1) + np.log(self.pi[k] + 1e-300)

            log_prob -= logsumexp(log_prob, axis=1, keepdims=True)
            post = np.exp(log_prob)  # (N, K)

            # M-step
            Nk = post.sum(axis=0)

            # Update pi
            self.pi = Nk / N

            # Update beta (weighted least squares per class)
            for k in range(K):
                if Nk[k] < 1:
                    continue
                W = np.diag(post[:, k])  # (N, N)
                X_design = np.tile(X, (N, 1))  # (N*T, P)
                y_vec = Y.flatten()  # (N*T,)
                w_vec = np.repeat(post[:, k], T)  # (N*T,)
                Xw = X_design * w_vec[:, np.newaxis]
                try:
                    self.beta[k] = np.linalg.lstsq(Xw, y_vec * w_vec, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass

            # Update sigma2
            for k in range(K):
                if Nk[k] < 1:
                    continue
                mu_k = X @ self.beta[k]
                resid = Y - mu_k[np.newaxis, :]
                w_resid = resid * post[:, k][:, np.newaxis]
                self.sigma2[k] = max(np.sum(w_resid ** 2) / (Nk[k] * T), 0.01)

        # Store results
        self.posterior = post
        self.labels = post.argmax(axis=1)
        self.log_likelihood = np.sum(logsumexp(log_prob, axis=1))

        # BIC / AIC
        n_params = K * (self.poly_order + 1) + K + (K - 1)  # beta + sigma2 + pi
        self.bic = -2 * self.log_likelihood + n_params * np.log(N)
        self.aic = -2 * self.log_likelihood + 2 * n_params
        self.n_params = n_params

        # Entropy R^2
        entropy = -np.sum(post * np.log(post + 1e-300))
        entropy_max = -N * np.log(K)
        self.entropy_r2 = 1 - entropy / entropy_max if entropy_max != 0 else 0

        # Average posterior probability per class
        self.avg_post = np.zeros(K)
        for k in range(K):
            mask = self.labels == k
            if mask.sum() > 0:
                self.avg_post[k] = post[mask, k].mean()

    def predict_proba(self, Y_new):
        N_new = Y_new.shape[0]
        K = self.n_classes
        X = self.X
        log_prob = np.zeros((N_new, K))
        for k in range(K):
            mu_k = X @ self.beta[k]
            resid = Y_new - mu_k[np.newaxis, :]
            log_pdf = -0.5 * np.log(self.sigma2[k]) - 0.5 * resid ** 2 / self.sigma2[k]
            log_prob[:, k] = np.sum(log_pdf, axis=1) + np.log(self.pi[k] + 1e-300)
        log_prob -= logsumexp(log_prob, axis=1, keepdims=True)
        return np.exp(log_prob)

    def get_trajectory_means(self):
        K = self.n_classes
        means = np.zeros((K, self.T))
        for k in range(K):
            means[k] = self.X @ self.beta[k]
        return means


# ── 2a. 对每个 core marker 做 GBTM ────────────────────────
univariate_results = {}

for marker in TRAJ_MARKERS:
    print(f"\n--- {marker.upper()} ---")
    cols = [f"{marker}_{w}" for w in WINDOWS]
    Y = df_traj[cols].values.astype(float)

    mask_complete = np.all(np.isfinite(Y), axis=1)
    Y_clean = Y[mask_complete]
    print(f"  Complete cases: {mask_complete.sum()}/{len(Y)}")

    best_model = None
    best_bic = np.inf
    model_comparison = []

    for n_cls in range(2, 6):
        try:
            model = GBTMUnivariate(
                n_classes=n_cls, polynomial_order=1,
                n_init=30, max_iter=500, random_state=42
            )
            model.fit(Y_clean, time_points=np.array(WINDOW_MIDS))

            min_class_pct = min(model.pi) * 100
            min_avg_post = min(model.avg_post)

            model_comparison.append({
                "n_classes": n_cls,
                "BIC": model.bic,
                "AIC": model.aic,
                "entropy_r2": model.entropy_r2,
                "min_class_pct": round(min_class_pct, 1),
                "min_avg_post": round(min_avg_post, 3),
                "log_likelihood": model.log_likelihood,
            })

            print(f"  K={n_cls}: BIC={model.bic:.1f}, AIC={model.aic:.1f}, "
                  f"Entropy R2={model.entropy_r2:.3f}, "
                  f"min_class={min_class_pct:.1f}%, min_avg_post={min_avg_post:.3f}")

            if model.bic < best_bic and min_class_pct >= 5.0:
                best_bic = model.bic
                best_model = model
        except Exception as e:
            print(f"  K={n_cls}: FAILED - {e}")

    if best_model is not None:
        univariate_results[marker] = {
            "model": best_model,
            "comparison": pd.DataFrame(model_comparison),
            "Y_clean": Y_clean,
            "mask": mask_complete,
        }
        print(f"  >>> Best: K={best_model.n_classes}, BIC={best_model.bic:.1f}")
    else:
        print(f"  >>> No valid model for {marker}")

# ═══════════════════════════════════════════════════════════
# 3. 多变量联合 GBTM（核心产出）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 3: Multivariate Joint GBTM")
print("=" * 70)

# ── 3a. 构建联合轨迹矩阵 ──────────────────────────────────
# 对每个 marker 标准化后拼接，形成 N × (T × M) 矩阵
# GBTM 在标准化空间中聚类，然后反变换回原始尺度

scalers = {}
Y_standardized = np.zeros((len(df_traj), len(TRAJ_MARKERS) * len(WINDOWS)))

for i, marker in enumerate(TRAJ_MARKERS):
    cols = [f"{marker}_{w}" for w in WINDOWS]
    Y_raw = df_traj[cols].values.astype(float)
    scaler = StandardScaler()
    Y_std = scaler.fit_transform(Y_raw)
    scalers[marker] = scaler
    Y_standardized[:, i * len(WINDOWS):(i + 1) * len(WINDOWS)] = Y_std

mask_all_complete = np.all(np.isfinite(Y_standardized), axis=1)
Y_joint = Y_standardized[mask_all_complete]
print(f"Joint trajectory matrix: {Y_joint.shape[0]} complete cases, {Y_joint.shape[1]} features")

# ── 3b. 多变量 GBTM ───────────────────────────────────────
# 使用 GMM 作为 GBTM 的快速近似（N=483 足够）
# 对联合特征矩阵做 GMM，等价于多变量 GBTM

joint_model_comparison = []
best_joint_model = None
best_joint_bic = np.inf

for n_cls in range(2, 6):
    try:
        gmm = GaussianMixture(
            n_components=n_cls,
            n_init=50,
            max_iter=1000,
            random_state=42,
            covariance_type="full",
            reg_covar=1e-5,
        )
        gmm.fit(Y_joint)

        labels = gmm.predict(Y_joint)
        post = gmm.predict_proba(Y_joint)

        class_sizes = pd.Series(labels).value_counts()
        min_class_pct = class_sizes.min() / len(labels) * 100

        avg_post_per_class = []
        for k in range(n_cls):
            mask_k = labels == k
            if mask_k.sum() > 0:
                avg_post_per_class.append(post[mask_k, k].mean())
            else:
                avg_post_per_class.append(0)
        min_avg_post = min(avg_post_per_class)

        entropy = -np.sum(post * np.log(post + 1e-300))
        entropy_max = -len(Y_joint) * np.log(n_cls)
        entropy_r2 = 1 - entropy / entropy_max if entropy_max != 0 else 0

        joint_model_comparison.append({
            "n_classes": n_cls,
            "BIC": gmm.bic(Y_joint),
            "AIC": gmm.aic(Y_joint),
            "entropy_r2": round(entropy_r2, 3),
            "min_class_pct": round(min_class_pct, 1),
            "min_avg_post": round(min_avg_post, 3),
            "log_likelihood": gmm.score(Y_joint) * len(Y_joint),
            "class_sizes": str(class_sizes.sort_index().tolist()),
        })

        print(f"  K={n_cls}: BIC={gmm.bic(Y_joint):.1f}, AIC={gmm.aic(Y_joint):.1f}, "
              f"Entropy R2={entropy_r2:.3f}, "
              f"min_class={min_class_pct:.1f}%, min_avg_post={min_avg_post:.3f}, "
              f"sizes={class_sizes.sort_index().tolist()}")

        if gmm.bic(Y_joint) < best_joint_bic and min_class_pct >= 5.0 and min_avg_post >= 0.60:
            best_joint_bic = gmm.bic(Y_joint)
            best_joint_model = gmm
    except Exception as e:
        print(f"  K={n_cls}: FAILED - {e}")

df_model_comp = pd.DataFrame(joint_model_comparison)
df_model_comp.to_csv(os.path.join(OUTDIR, "gbtm_model_comparison.csv"), index=False)
print(f"\n  Model comparison saved to gbtm_model_comparison.csv")

if best_joint_model is None:
    print("\n  [FATAL] No valid joint GBTM model found. Check data.")
    sys.exit(1)

# ── 3c. 提取最优模型的轨迹分配 ─────────────────────────────
K_opt = best_joint_model.n_components
joint_labels = best_joint_model.predict(Y_joint)
joint_post = best_joint_model.predict_proba(Y_joint)

print(f"\n  Optimal K = {K_opt}")
print(f"  Class distribution:")
for k in range(K_opt):
    n_k = (joint_labels == k).sum()
    print(f"    Class {k}: n={n_k} ({100*n_k/len(joint_labels):.1f}%), "
          f"avg_post={joint_post[joint_labels==k, k].mean():.3f}")

# ── 3d. 计算每类在原始尺度上的轨迹均值 ─────────────────────
class_profiles = []
for k in range(K_opt):
    mask_k = joint_labels == k
    for marker in TRAJ_MARKERS:
        for j, w in enumerate(WINDOWS):
            col = f"{marker}_{w}"
            vals = df_traj.loc[mask_all_complete, col].values[mask_k]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                class_profiles.append({
                    "class": k,
                    "marker": marker,
                    "window": w,
                    "time_mid": WINDOW_MIDS[j],
                    "mean": round(vals.mean(), 2),
                    "std": round(vals.std(), 2),
                    "median": round(np.median(vals), 2),
                    "q25": round(np.percentile(vals, 25), 2),
                    "q75": round(np.percentile(vals, 75), 2),
                    "n": len(vals),
                })

df_profiles = pd.DataFrame(class_profiles)
df_profiles.to_csv(os.path.join(OUTDIR, "gbtm_class_profiles.csv"), index=False)
print(f"\n  Class profiles saved to gbtm_class_profiles.csv")

# ═══════════════════════════════════════════════════════════
# 4. 轨迹类命名与临床解释
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 4: Trajectory Class Characterization")
print("=" * 70)

# 基于轨迹斜率（w6_24 → w24_48 变化方向）命名
class_names = {}
class_slopes = {}

for k in range(K_opt):
    mask_k = joint_labels == k
    slopes = {}
    for marker in TRAJ_MARKERS:
        v1 = df_traj.loc[mask_all_complete, f"{marker}_w6_24"].values[mask_k]
        v2 = df_traj.loc[mask_all_complete, f"{marker}_w24_48"].values[mask_k]
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        if len(v1) > 0 and len(v2) > 0:
            slopes[marker] = v2.mean() - v1.mean()
        else:
            slopes[marker] = 0
    class_slopes[k] = slopes

    cr_slope = slopes.get("creatinine", 0)
    bun_slope = slopes.get("bun", 0)
    wbc_slope = slopes.get("wbc", 0)
    plt_slope = slopes.get("platelet", 0)
    glu_slope = slopes.get("glucose", 0)

    n_organ_worsening = sum(1 for m, s in [("creatinine", cr_slope), ("bun", bun_slope)]
                            if s > 0)
    n_inflam_worsening = sum(1 for s in [wbc_slope, glu_slope] if s > 0)
    platelet_falling = plt_slope < -5

    if cr_slope > 2.0 or (cr_slope > 0.5 and bun_slope > 0):
        class_names[k] = "Renal-Deterioration"
    elif platelet_falling and n_inflam_worsening >= 1:
        class_names[k] = "Inflammatory-Deterioration"
    elif n_organ_worsening == 0 and n_inflam_worsening == 0 and not platelet_falling:
        class_names[k] = "Rapid-Recovery"
    else:
        class_names[k] = "Slow-Recovery"

    print(f"\n  Class {k} → {class_names[k]} (n={mask_k.sum()})")
    for m, s in slopes.items():
        direction = "↓" if s < 0 else "↑" if s > 0 else "→"
        print(f"    {m}: Δ = {s:+.2f} {direction}")

# ═══════════════════════════════════════════════════════════
# 5. 轨迹类 × 结局交叉表
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 5: Trajectory Class × Outcomes")
print("=" * 70)

outcomes = ["icu_flag", "icu_7d", "death_28d", "hospital_expire_flag"]
outcome_labels = ["ICU (any)", "ICU (7d)", "Death (28d)", "Hospital mortality"]

outcome_table = []
for k in range(K_opt):
    mask_k = joint_labels == k
    n_k = mask_k.sum()
    row = {"class": k, "class_name": class_names[k], "n": n_k}
    for out, out_label in zip(outcomes, outcome_labels):
        vals = df_traj.loc[mask_all_complete, out].values[mask_k]
        n_event = vals.sum()
        pct = 100 * n_event / n_k if n_k > 0 else 0
        row[f"{out_label}_n"] = int(n_event)
        row[f"{out_label}_pct"] = round(pct, 1)
    # LOS
    los_vals = df_traj.loc[mask_all_complete, "los_days"].values[mask_k]
    los_vals = los_vals[np.isfinite(los_vals)]
    row["LOS_median"] = round(np.median(los_vals), 1) if len(los_vals) > 0 else None
    row["LOS_iqr"] = f"{np.percentile(los_vals, 25):.1f}-{np.percentile(los_vals, 75):.1f}" if len(los_vals) > 0 else None
    outcome_table.append(row)

df_outcome = pd.DataFrame(outcome_table)
df_outcome.to_csv(os.path.join(OUTDIR, "gbtm_class_outcome_table.csv"), index=False)
print(df_outcome.to_string(index=False))

# Fisher exact test for ICU by class
if K_opt >= 2:
    contingency = np.zeros((K_opt, 2))
    for k in range(K_opt):
        mask_k = joint_labels == k
        vals = df_traj.loc[mask_all_complete, "icu_flag"].values[mask_k]
        contingency[k, 0] = (vals == 0).sum()
        contingency[k, 1] = vals.sum()
    try:
        from scipy.stats import chi2_contingency
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        print(f"\n  Chi-squared test (ICU by class): chi2={chi2:.2f}, df={dof}, p={p_val:.4f}")
    except Exception as e:
        print(f"\n  Chi-squared test failed: {e}")

# ═══════════════════════════════════════════════════════════
# 6. 轨迹图
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 6: Trajectory Visualization")
print("=" * 70)

COLORS = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"]
CLASS_COLORS = COLORS[:K_opt]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("AI-FAP: 48h Inflammatory-Organ Function Trajectories (GBTM)",
             fontsize=14, fontweight="bold", y=0.98)

for idx, marker in enumerate(TRAJ_MARKERS):
    ax = axes[idx // 3, idx % 3]
    for k in range(K_opt):
        mask_k = joint_labels == k
        means = []
        cis_low = []
        cis_high = []
        for j, w in enumerate(WINDOWS):
            col = f"{marker}_{w}"
            vals = df_traj.loc[mask_all_complete, col].values[mask_k]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 1:
                means.append(vals.mean())
                se = vals.std() / np.sqrt(len(vals))
                cis_low.append(vals.mean() - 1.96 * se)
                cis_high.append(vals.mean() + 1.96 * se)
            else:
                means.append(np.nan)
                cis_low.append(np.nan)
                cis_high.append(np.nan)

        ax.plot(WINDOW_MIDS, means, "o-", color=CLASS_COLORS[k],
                label=f"{class_names[k]} (n={mask_k.sum()})",
                linewidth=2, markersize=8)
        ax.fill_between(WINDOW_MIDS, cis_low, cis_high,
                        color=CLASS_COLORS[k], alpha=0.15)

    ax.set_title(marker.upper(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Hours from admission")
    ax.set_ylabel(marker.upper())
    ax.set_xticks(WINDOW_MIDS)
    ax.set_xticklabels(WINDOW_LABELS)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

# 第 6 子图：综合摘要（类大小饼图 + 结局条形图）
ax_summary = axes[1, 2]
class_sizes = [int((joint_labels == k).sum()) for k in range(K_opt)]
class_labels_pie = [f"{class_names[k]}\nn={class_sizes[k]}" for k in range(K_opt)]
wedges, texts, autotexts = ax_summary.pie(
    class_sizes, labels=class_labels_pie, colors=CLASS_COLORS,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9}
)
ax_summary.set_title("Class Distribution", fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = os.path.join(OUTDIR, "gbtm_trajectory_plot.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Trajectory plot saved to {fig_path}")

# ── 6b. 结局对比条形图 ─────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
fig2.suptitle("AI-FAP: Outcomes by GBTM Trajectory Class",
              fontsize=13, fontweight="bold")

for idx, (out, out_label) in enumerate(zip(outcomes, outcome_labels)):
    ax = axes2[idx]
    pcts = []
    for k in range(K_opt):
        mask_k = joint_labels == k
        vals = df_traj.loc[mask_all_complete, out].values[mask_k]
        pcts.append(100 * vals.sum() / mask_k.sum() if mask_k.sum() > 0 else 0)

    bars = ax.bar(range(K_opt), pcts, color=CLASS_COLORS, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(K_opt))
    ax.set_xticklabels([class_names[k] for k in range(K_opt)], fontsize=8, rotation=15)
    ax.set_ylabel("%")
    ax.set_title(out_label, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig2_path = os.path.join(OUTDIR, "gbtm_outcome_comparison.png")
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Outcome comparison plot saved to {fig2_path}")

# ═══════════════════════════════════════════════════════════
# 7. 保存轨迹分配
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 7: Save Trajectory Assignments")
print("=" * 70)

assignment_rows = []
eligible_hadm = df_traj.loc[mask_all_complete, "hadm_id"].values

for i, hadm_id in enumerate(eligible_hadm):
    row = {"hadm_id": int(hadm_id), "gbtm_class": int(joint_labels[i])}
    row["gbtm_class_name"] = class_names[joint_labels[i]]
    for k in range(K_opt):
        row[f"post_class_{k}"] = round(joint_post[i, k], 4)
    row["max_posterior"] = round(joint_post[i].max(), 4)
    assignment_rows.append(row)

df_assign = pd.DataFrame(assignment_rows)
assign_path = os.path.join(OUTDIR, "gbtm_trajectory_assignments.csv")
df_assign.to_csv(assign_path, index=False)
print(f"  Assignments saved to {assign_path} ({len(df_assign)} rows)")

# ── 7b. 合并回完整队列（ ineligible 标记为 NaN） ──────────
df_full = df.copy()
df_full = df_full.merge(df_assign, on="hadm_id", how="left")
full_assign_path = os.path.join(OUTDIR, "gbtm_assignments_full_cohort.csv")
df_full.to_csv(full_assign_path, index=False)
print(f"  Full cohort with assignments saved to {full_assign_path}")

# ═══════════════════════════════════════════════════════════
# 8. 地标特征工程（为 03_landmark_ml.py 准备）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 8: Landmark Feature Engineering")
print("=" * 70)

# ── 8a. T0+24h 地标：基线 + w0_6 + w6_24 ──────────────────
# ── 8b. T0+48h 地标：基线 + w0_6 + w6_24 + w24_48 ────────

BASELINE_COLS = [
    "age", "gender",
    "tg_ge500_flag", "metabolic_dx_flag", "tg_admission",
    "diabetes", "hypertension", "heart_failure", "cad", "copd", "ckd",
    "dyslipidemia", "obesity_dx", "htg_dx",
    "baseline_wbc", "baseline_lactate", "baseline_creatinine", "baseline_bun",
    "baseline_bilirubin", "baseline_platelet", "baseline_glucose",
    "baseline_lipase", "baseline_amylase", "baseline_calcium",
    "baseline_alt", "baseline_ast", "baseline_tg",
]

# 动态特征：斜率（变化率）
def compute_dynamic_features(df_in, markers, windows_available):
    """Compute trajectory-derived dynamic features."""
    result = df_in[["hadm_id"]].copy()

    for marker in markers:
        avail_w = [w for w in windows_available if f"{marker}_{w}" in df_in.columns]
        if len(avail_w) >= 2:
            first_w = avail_w[0]
            last_w = avail_w[-1]
            col_first = f"{marker}_{first_w}"
            col_last = f"{marker}_{last_w}"
            result[f"{marker}_slope"] = df_in[col_last] - df_in[col_first]
            result[f"{marker}_delta_pct"] = (
                (df_in[col_last] - df_in[col_first]) / (df_in[col_first] + 1e-6) * 100
            )
            result[f"{marker}_max"] = df_in[[f"{marker}_{w}" for w in avail_w]].max(axis=1)
            result[f"{marker}_min"] = df_in[[f"{marker}_{w}" for w in avail_w]].min(axis=1)
            result[f"{marker}_range"] = result[f"{marker}_max"] - result[f"{marker}_min"]

        for w in avail_w:
            col = f"{marker}_{w}"
            result[col] = df_in[col]

    return result


ALL_WINDOWS = ["w0_6", "w6_24", "w24_48"]
ALL_MARKERS_FEAT = TRAJ_MARKERS + OPTIONAL_MARKERS

# T0+24h landmark: windows w0_6 + w6_24
landmark_24_markers = ALL_MARKERS_FEAT
landmark_24_windows = ["w0_6", "w6_24"]
feat_24 = compute_dynamic_features(df, landmark_24_markers, landmark_24_windows)

# Add baseline
for col in BASELINE_COLS:
    if col in df.columns:
        feat_24[col] = df[col]

# Add outcome
for out in outcomes + ["los_days"]:
    feat_24[out] = df[out]

feat_24_path = os.path.join(OUTDIR, "landmark_features_24h.csv")
feat_24.to_csv(feat_24_path, index=False)
print(f"  24h landmark features: {feat_24.shape} -> {feat_24_path}")

# T0+48h landmark: all three windows
feat_48 = compute_dynamic_features(df, ALL_MARKERS_FEAT, ALL_WINDOWS)

for col in BASELINE_COLS:
    if col in df.columns:
        feat_48[col] = df[col]

for out in outcomes + ["los_days"]:
    feat_48[out] = df[out]

# Add GBTM class assignment
feat_48 = feat_48.merge(df_assign[["hadm_id", "gbtm_class", "gbtm_class_name"]],
                         on="hadm_id", how="left")

feat_48_path = os.path.join(OUTDIR, "landmark_features_48h.csv")
feat_48.to_csv(feat_48_path, index=False)
print(f"  48h landmark features: {feat_48.shape} -> {feat_48_path}")

# ═══════════════════════════════════════════════════════════
# 9. 敏感性分析：w0_6 纳入 vs 不纳入
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 9: Sensitivity — Including w0_6 Window")
print("=" * 70)

# 三窗版 GBTM（w0_6 覆盖率低，需更宽松的完整度要求）
THREE_WINDOWS = ["w0_6", "w6_24", "w24_48"]
THREE_MIDS = [3.0, 15.0, 36.0]

df_traj_3w = df.copy()
MIN_CORE_3W = 3  # 放宽：至少 3/5 core markers 三窗均完整

def check_3w_completeness(row):
    complete = 0
    for m in TRAJ_MARKERS:
        has_all = all(pd.notna(row[f"{m}_{w}"]) for w in THREE_WINDOWS)
        if has_all:
            complete += 1
    return complete

df_traj_3w["traj_3w_complete"] = df_traj_3w.apply(check_3w_completeness, axis=1)
df_traj_3w["traj_3w_eligible"] = df_traj_3w["traj_3w_complete"] >= MIN_CORE_3W
n_3w = df_traj_3w["traj_3w_eligible"].sum()
print(f"  3-window eligible: {n_3w}/{len(df_traj_3w)} ({100*n_3w/len(df_traj_3w):.1f}%)")

if n_3w >= 100:
    df_3w = df_traj_3w[df_traj_3w["traj_3w_eligible"]].copy().reset_index(drop=True)

    Y_3w_std = np.zeros((len(df_3w), len(TRAJ_MARKERS) * 3))
    for i, marker in enumerate(TRAJ_MARKERS):
        cols = [f"{marker}_{w}" for w in THREE_WINDOWS]
        Y_raw = df_3w[cols].values.astype(float)
        scaler = StandardScaler()
        Y_std = scaler.fit_transform(Y_raw)
        Y_3w_std[:, i * 3:(i + 1) * 3] = Y_std

    mask_3w = np.all(np.isfinite(Y_3w_std), axis=1)
    Y_3w_clean = Y_3w_std[mask_3w]
    print(f"  3-window complete cases: {Y_3w_clean.shape[0]}")

    if Y_3w_clean.shape[0] >= 80:
        sens_results = []
        for n_cls in range(2, 5):
            try:
                gmm = GaussianMixture(n_components=n_cls, n_init=30,
                                       max_iter=500, random_state=42,
                                       covariance_type="full", reg_covar=1e-5)
                gmm.fit(Y_3w_clean)
                labels = gmm.predict(Y_3w_clean)
                min_pct = pd.Series(labels).value_counts().min() / len(labels) * 100
                sens_results.append({
                    "n_classes": n_cls,
                    "BIC": gmm.bic(Y_3w_clean),
                    "AIC": gmm.aic(Y_3w_clean),
                    "min_class_pct": round(min_pct, 1),
                    "n_complete": Y_3w_clean.shape[0],
                })
                print(f"  3w K={n_cls}: BIC={gmm.bic(Y_3w_clean):.1f}, "
                      f"min_class={min_pct:.1f}%")
            except Exception as e:
                print(f"  3w K={n_cls}: FAILED - {e}")

        if sens_results:
            df_sens = pd.DataFrame(sens_results)
            df_sens.to_csv(os.path.join(OUTDIR, "gbtm_sensitivity_3window.csv"), index=False)
            print(f"  Sensitivity results saved")
    else:
        print(f"  Too few complete cases for 3-window GBTM ({Y_3w_clean.shape[0]})")
else:
    print(f"  Too few eligible for 3-window analysis ({n_3w})")

# ═══════════════════════════════════════════════════════════
# 10. 汇总输出
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
GBTM Trajectory Modeling Complete
─────────────────────────────────
Input cohort:       {len(df)} MDAP admissions
GBTM-eligible:      {len(df_traj)} (>= {MIN_CORE_COMPLETE}/5 core markers, 2-window)
Joint model:        K={K_opt} classes
Class names:        {', '.join(f'{k}:{class_names[k]}' for k in range(K_opt))}
Class sizes:        {', '.join(f'{(joint_labels==k).sum()}' for k in range(K_opt))}

Output files:
  gbtm_model_comparison.csv      - 2-5 class model comparison
  gbtm_class_profiles.csv        - per-class trajectory means/CI
  gbtm_trajectory_assignments.csv - per-hadm_id class + posterior
  gbtm_assignments_full_cohort.csv - full cohort with class (NaN for ineligible)
  gbtm_class_outcome_table.csv   - class × outcomes
  gbtm_trajectory_plot.png       - trajectory visualization
  gbtm_outcome_comparison.png    - outcome bar chart
  landmark_features_24h.csv      - T0+24h feature set
  landmark_features_48h.csv      - T0+48h feature set + GBTM class
""")

print("Next: 03_landmark_ml.py — Landmark ML dynamic restratification")
