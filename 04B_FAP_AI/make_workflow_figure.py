"""Figure 5 - AI-FAP two-stage clinical workflow schematic (JBI submission)."""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
})

BASE = os.path.dirname(os.path.abspath(__file__))
OUTS = [os.path.normpath(os.path.join(BASE, "..", "manuscript", "JBI", "figures")),
        os.path.normpath(os.path.join(BASE, "..", "manuscript", "latex", "figures"))]
for d in OUTS:
    os.makedirs(d, exist_ok=True)

NAVY, BLUE, GREY = "#22405f", "#3a78b5", "#eef2f6"
RED, GREEN, AMBER = "#c0392b", "#2e7d52", "#b9770e"
MM = 1 / 25.4

fig, ax = plt.subplots(figsize=(183 * MM, 118 * MM))
ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, txt, fc, ec, fs=6.6, tc="#1a1a1a", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.1",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, color=NAVY, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=color))


# ---- Stage banners ----
ax.text(1, 63.5, "Stage 1  |  Admission (T0): deployable triage layer", ha="left", va="center",
        fontsize=8, fontweight="bold", color=NAVY)
ax.text(1, 30.5, "Stage 2  |  First 0-48 h: trajectory-informed subtype layer (monitoring aid)",
        ha="left", va="center", fontsize=8, fontweight="bold", color=NAVY)
ax.add_patch(plt.Rectangle((0.5, 33.5), 99, 26.8, fill=False, ec="#c9d4df", lw=0.8, ls=(0, (4, 3))))
ax.add_patch(plt.Rectangle((0.5, 1.2), 99, 28.0, fill=False, ec="#c9d4df", lw=0.8, ls=(0, (4, 3))))

# ---- Stage 1 chain ----
box(2, 44, 20, 11,
    "Admission inputs\n(demographics, metabolic\nphenotype, comorbidities,\nadmission labs)", GREY, NAVY)
box(27, 44, 20, 11,
    "Leakage-corrected\nadmission model\n+ uncertainty", "#e8f0f7", BLUE, bold=False)
arrow(22, 49.5, 27, 49.5)

# Stage 1 triage outputs (3 chips)
box(53, 50.5, 45, 6.0,
    "Routine-monitoring (event rate 19.8%) -> standard ward observation", "#e9f0e9", GREEN, fs=6.3)
box(53, 43.7, 45, 6.0,
    "Abstention / low confidence (27.6%) -> mandatory senior review (not de-escalation)",
    "#fff4e0", AMBER, fs=6.3)
box(53, 36.9, 45, 6.0,
    "High-risk -> intensified surveillance / early ICU consideration", "#fbe9e7", RED, fs=6.3)
for yy in (53.5, 46.7, 39.9):
    arrow(47, 49.5, 53, yy, color="#8a98a6", lw=1.0)

# ---- link Stage 1 high-risk -> Stage 2 ----
arrow(75.5, 36.9, 75.5, 26.2, color=RED, lw=1.6)
ax.text(77, 31.5, "high-risk group", ha="left", va="center", fontsize=5.8, color=RED, style="italic")

# ---- Stage 2 chain ----
box(2, 14, 22, 11,
    "Observed 0-48 h\nmetabolic-inflammatory\ntrajectory (GBTM)\nread, not predicted", GREY, NAVY)
box(53, 19.2, 45, 6.0,
    "Metabolic-vulnerable (50.0%) -> glucose, TG clearance, calcium, acid-base", "#eef0f7", BLUE, fs=6.2)
box(53, 12.4, 45, 6.0,
    "Inflammatory-vulnerable (83.3%) -> organ function, systemic inflammation", "#f2e9f3", "#7d4b86", fs=6.2)
box(53, 5.6, 45, 6.0,
    "Dual-vulnerable (100%) -> both axes; lowest escalation threshold", "#fbe3df", RED, fs=6.2)
box(27, 14, 20, 11, "Subtype the\nhigh-risk group\n(mechanism of\ndeterioration)", "#e8f0f7", BLUE)
arrow(24, 19.5, 27, 19.5)
for yy in (22.2, 15.4, 8.6):
    arrow(47, 19.5, 53, yy, color="#8a98a6", lw=1.0)

# ---- boundary footer ----
ax.text(50, 1.9,
        "Output: surveillance-priority strata + senior-review triggers in the EHR. "
        "Not a treatment recommendation; subtype is read from accrued 0-48 h observed data; "
        "it is not an admission-time prediction; "
        "local recalibration and prospective silent testing required before deployment.",
        ha="center", va="center", fontsize=5.6, color="#444444", wrap=True)

for d in OUTS:
    fig.savefig(os.path.join(d, "figure5_workflow.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(d, "figure5_workflow.png"), dpi=600, bbox_inches="tight")
plt.close(fig)
print("-> figure5_workflow")
