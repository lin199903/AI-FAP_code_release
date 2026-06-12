# -*- coding: utf-8 -*-
"""
JBI 论文发表级图（R1-R4）。Python/matplotlib，矢量 PDF + PNG。
读 04B_FAP_AI/outputs 的 CSV，输出到 manuscript/latex/figures/。
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIRS = [
    os.path.normpath(os.path.join(BASE, "..", "manuscript", "latex", "figures")),
    os.path.normpath(os.path.join(BASE, "..", "manuscript", "JBI", "figures")),
]
for fig_dir in FIG_DIRS:
    os.makedirs(fig_dir, exist_ok=True)

NAVY, BLUE, GREY = "#22405f", "#3a78b5", "#9aa3ab"
RED, GREEN, AMBER = "#c0392b", "#2e7d52", "#d39a2d"
MM = 1 / 25.4

FEATURE_LABELS = {
    "tg_ge500_flag": "TG >=500\nmg/dL",
    "cad": "Coronary artery\ndisease",
    "obesity_dx": "Obesity diagnosis",
    "htg_dx": "Hypertriglyceridaemia\ndiagnosis",
    "diabetes": "Diabetes",
    "age": "Age",
    "baseline_calcium": "Calcium",
    "baseline_lipase": "Lipase",
    "baseline_glucose": "Glucose",
    "baseline_bilirubin": "Bilirubin",
    "baseline_wbc": "WBC",
    "baseline_bun": "BUN",
    "metabolic_dx_flag": "Metabolic diagnosis",
}


def feature_label(feature):
    return FEATURE_LABELS.get(feature, feature.replace("baseline_", "").replace("_", " ").title())


def save(fig, name):
    for fig_dir in FIG_DIRS:
        fig.savefig(os.path.join(fig_dir, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


def panel_tag(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


# ════════════════════ Figure 1 — R1 front-loading ════════════════════
def fig1():
    dt = pd.read_csv(os.path.join(OUT, "deterioration_timing.csv"))
    evt = dt.loc[dt["composite"] == 1, "evt_time_h"].dropna().values
    evt = np.sort(evt)
    n = len(evt)
    cum = np.arange(1, n + 1) / n * 100

    fig = plt.figure(figsize=(183 * MM, 70 * MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.32, width_ratios=[1.25, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.step(np.concatenate([[0], evt]), np.concatenate([[0], cum]), where="post",
            color=NAVY, lw=1.6)
    for h, lab in [(24, "24 h"), (48, "48 h")]:
        pct = 100 * (evt <= h).mean()
        ax.axvline(h, color=GREY, lw=0.7, ls="--")
        ax.plot(h, pct, "o", color=RED, ms=4, zorder=5)
        ax.annotate(f"{pct:.0f}% by {lab}", (h, pct), textcoords="offset points",
                    xytext=(6, -2), fontsize=6.4, color=RED)
    ax.set_xlim(0, 168); ax.set_ylim(0, 102)
    ax.set_xlabel("Hours after admission"); ax.set_ylabel("Cumulative events (%)")
    ax.set_title("Early escalation is front-loaded", fontsize=8)
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
    panel_tag(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    icu = dt.loc[dt["icu_7d"] == 1, "icu_intime_hours"].dropna().values
    bins = [0, 6, 24, 48, 168]
    labels = ["0-6", "6-24", "24-48", "48-168"]
    counts = [((icu > lo) & (icu <= hi)).sum() for lo, hi in zip(bins[:-1], bins[1:])]
    counts[0] = (icu <= 6).sum()
    pct = 100 * np.array(counts) / len(icu)
    cols = [RED, AMBER, GREY, GREY]
    ax.bar(range(4), pct, color=cols, edgecolor="black", lw=0.5, width=0.72)
    for i, (c, p) in enumerate(zip(counts, pct)):
        ax.text(i, p + 1.5, f"{p:.0f}%\n(n={c})", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylim(0, max(pct) + 14)
    ax.set_xlabel("ICU transfer time after admission (h)")
    ax.set_ylabel("Share of ICU transfers (%)")
    ax.set_title("Two-thirds of ICU within 6 h", fontsize=8)
    panel_tag(ax, "b")
    save(fig, "figure1_frontloading")


# ════════════════════ Figure 2 — R2 admission model ════════════════════
def fig2():
    pr = pd.read_csv(os.path.join(OUT, "oof_predictions_corrected.csv"))
    y = pr["composite_outcome"].values
    p = pr["prob_oof"].values
    sh = pd.read_csv(os.path.join(OUT, "landmark_ml_shap_T0.csv")).head(10)

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.52)

    # a ROC
    ax = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    ax.plot(fpr, tpr, color=NAVY, lw=1.6, label=f"OOF AUROC 0.735\n(95% CI 0.718-0.748)\nperm. p=0.001")
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--")
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Admission (T0) discrimination (repeated-CV OOF)", fontsize=8)
    ax.legend(loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel_tag(ax, "a")

    # b calibration
    ax = fig.add_subplot(gs[0, 1])
    pt, pp = calibration_curve(y, p, n_bins=6, strategy="quantile")
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--")
    ax.plot(pp, pt, "o-", color=BLUE, lw=1.4, ms=4)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (OOF slope 0.85)", fontsize=8)
    ax.set_xlim(0, 0.85); ax.set_ylim(0, 0.85)
    panel_tag(ax, "b")

    # c risk strata
    ax = fig.add_subplot(gs[1, 0])
    cats = ["Low", "Intermediate", "High"]
    rates, ns = [], []
    for c in cats:
        m = pr["risk_category"] == c
        ns.append(int(m.sum()))
        rates.append(100 * pr.loc[m, "composite_outcome"].mean() if m.sum() else 0)
    cols = [GREEN, AMBER, RED]
    ax.bar(range(3), rates, color=cols, edgecolor="black", lw=0.5, width=0.66)
    for i, (r, nn) in enumerate(zip(rates, ns)):
        ax.text(i, r + 1.2, f"{r:.1f}%\n(n={nn})", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(range(3)); ax.set_xticklabels(cats)
    ax.set_ylim(0, max(rates) + 12); ax.set_ylabel("Event rate (%)")
    ax.set_title("Monotonic risk stratification", fontsize=8)
    panel_tag(ax, "c")

    # d SHAP
    ax = fig.add_subplot(gs[1, 1])
    feats = sh["feature"][::-1].values
    vals = sh["mean_abs_shap"][::-1].values
    ax.barh(range(len(feats)), vals, color=NAVY, height=0.7)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([feature_label(f) for f in feats], fontsize=5.8)
    ax.set_xlabel("Mean |SHAP|"); ax.set_title("Top admission predictors", fontsize=8)
    panel_tag(ax, "d")
    save(fig, "figure2_admission_model")


# ════════════════════ Figure 3 — R3 leakage correction ════════════════════
def fig3():
    perf = pd.read_csv(os.path.join(OUT, "landmark_ml_performance.csv"))
    perf = perf[perf["model"] == "LightGBM"].set_index("landmark")
    lms = ["T0", "T24", "T48"]
    pre = {"T0": 0.837, "T24": 0.900, "T48": 0.909}     # 泄漏修复前
    post = [perf.loc[lm, "auroc"] for lm in lms]
    nev = [int(perf.loc[lm, "n_events_val"]) for lm in lms]

    fig = plt.figure(figsize=(183 * MM, 68 * MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.34, width_ratios=[1.25, 1])

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, [pre[l] for l in lms], w, color=RED, alpha=0.85,
           edgecolor="black", lw=0.5, label="Before leakage correction")
    ax.bar(x + w / 2, post, w, color=NAVY, edgecolor="black", lw=0.5,
           label="After correction")
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    for i, l in enumerate(lms):
        ax.text(i - w / 2, pre[l] + 0.01, f"{pre[l]:.3f}", ha="center", fontsize=5.8, color=RED)
        ax.text(i + w / 2, post[i] + 0.01, f"{post[i]:.3f}", ha="center", fontsize=5.8, color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(["T0\n(admission)", "T24", "T48"])
    ax.set_ylim(0.45, 0.97); ax.set_ylabel("Validation AUROC")
    ax.set_title("Later landmarks become event-depleted", fontsize=8)
    ax.legend(loc="upper left")
    panel_tag(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    cols = [GREEN, GREY, GREY]
    ax.bar(x, nev, color=cols, edgecolor="black", lw=0.5, width=0.66)
    for i, nn in enumerate(nev):
        ax.text(i, nn + 0.6, str(nn), ha="center", fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels(["T0", "T24", "T48"])
    ax.set_ylim(0, max(nev) + 8); ax.set_ylabel("At-risk validation events (n)")
    ax.set_title("At-risk events depleted by 24-48 h", fontsize=8)
    panel_tag(ax, "b")
    save(fig, "figure3_leakage_correction")


# ════════════════════ Figure 4 — R4 governance ════════════════════
def fig4():
    cal = pd.read_csv(os.path.join(OUT, "governance_oof_calibration.csv"))
    ab = pd.read_csv(os.path.join(OUT, "governance_oof_abstention.csv"))
    sh = pd.read_csv(os.path.join(OUT, "eicu_transportability_shift.csv"))
    eicu_res = pd.read_csv(os.path.join(OUT, "eicu_transportability_results.csv"))
    eicu_tert = pd.read_csv(os.path.join(OUT, "eicu_tertile_enrichment.csv"))

    def get_metric(name):
        row = eicu_res.loc[eicu_res["metric"] == name, "value"]
        return float(row.iloc[0]) if len(row) else np.nan

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.36)

    # a recalibration
    ax = fig.add_subplot(gs[0, 0])
    sc = cal["scenario"].str.replace(" regression", "").str.replace(" scaling", "")
    x = np.arange(len(cal)); w = 0.38
    ax.bar(x - w / 2, cal["ece"], w, color=BLUE, edgecolor="black", lw=0.5, label="ECE")
    ax.bar(x + w / 2, cal["cal_slope"], w, color=AMBER, edgecolor="black", lw=0.5, label="Cal. slope")
    ax.axhline(1.0, color=GREY, lw=0.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(sc, fontsize=6)
    ax.set_ylabel("Value"); ax.set_title("Recalibration (ideal slope = 1)", fontsize=8)
    ax.legend(loc="upper right")
    panel_tag(ax, "a")

    # b abstention paradox
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ab["abstention_pct"], ab["auroc"], "o-", color=NAVY, lw=1.4, ms=3.5,
            label="Retained AUROC")
    ax.set_xlabel("Abstention (%)"); ax.set_ylabel("Retained AUROC", color=NAVY)
    ax.tick_params(axis="y", colors=NAVY)
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(ab["abstention_pct"], ab["outcome_rate_abstained"] * 100, "s--",
             color=RED, lw=1.3, ms=3.5, label="Abstained event rate")
    ax2.set_ylabel("Abstained event rate (%)", color=RED); ax2.tick_params(axis="y", colors=RED)
    ax.set_title("Abstention is not de-escalation (withheld sicker)", fontsize=8)
    panel_tag(ax, "b")

    # c eICU feature shift
    ax = fig.add_subplot(gs[1, 0])
    sh2 = sh.sort_values("smd")
    cols = [RED if f == "YES" else GREY for f in sh2["shift_flag"]]
    ax.barh(range(len(sh2)), sh2["smd"], color=cols, height=0.7)
    ax.axvline(0.2, color="black", lw=0.7, ls="--")
    ax.set_yticks(range(len(sh2)))
    ax.set_yticklabels([feature_label(f) for f in sh2["feature"]], fontsize=6)
    n_shift = int((sh["shift_flag"] == "YES").sum())
    ax.set_xlabel("Standardized mean difference")
    ax.set_title(f"MIMIC-IV vs eICU shift ({n_shift}/{len(sh)} > 0.2)", fontsize=8)
    panel_tag(ax, "c")

    # d eICU risk-tertile event rates (high-risk enrichment, descriptive)
    ax = fig.add_subplot(gs[1, 1])
    tert_order = ["Low", "Mid", "High"]
    tert_labels = {"Low": "Low", "Mid": "Intermediate", "High": "High"}
    eicu_tert = eicu_tert.set_index("tertile").reindex(tert_order).reset_index()
    tert = [tert_labels[t] for t in eicu_tert["tertile"]]
    erate = eicu_tert["event_rate_pct"].astype(float).tolist()
    ax.bar(range(3), erate, color=[GREEN, AMBER, RED], edgecolor="black", lw=0.5, width=0.62)
    for i, r in enumerate(erate):
        ax.text(i, r + 0.8, f"{r:.1f}%", ha="center", fontsize=6.4)
    ax.set_xticks(range(3)); ax.set_xticklabels(tert)
    ax.set_ylim(0, max(erate) + 12); ax.set_ylabel("eICU event rate (%)")
    slope = get_metric("Cal_slope")
    intercept = get_metric("Cal_intercept")
    ax.set_title(f"External: high-risk enrichment only\n(slope {slope:.2f}; intercept {intercept:.2f}, recal. needed)", fontsize=7.3)
    panel_tag(ax, "d")
    save(fig, "figure4_governance")


if __name__ == "__main__":
    print("Generating publication figures ->", "; ".join(FIG_DIRS))
    for fn in (fig1, fig2, fig3, fig4):
        try:
            fn()
        except Exception as e:
            print(f"  [{fn.__name__}] FAILED: {e}")
    print("done.")
