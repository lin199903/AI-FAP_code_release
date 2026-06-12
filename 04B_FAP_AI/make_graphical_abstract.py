"""Graphical abstract (Elsevier ~2.5:1) for the JBI submission."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
})

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(BASE, "..", "manuscript", "JBI", "figures"))
os.makedirs(FIG, exist_ok=True)

NAVY, BLUE, GREY = "#22405f", "#3a78b5", "#eef2f6"
RED, GREEN = "#c0392b", "#2e7d52"

fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")

ax.text(
    50,
    38.2,
    "Audit time-structure and prediction-time leakage before dynamic landmarking",
    ha="center",
    va="top",
    fontsize=9.2,
    fontweight="bold",
    color=NAVY,
)

stages = [
    (3, "Clinical question\nat admission\nsurveillance or\nsenior review", GREY, NAVY),
    (22, "Time-structure\naudit\n\n89% events <=48 h", "#e9f0e9", GREEN),
    (41, "Prediction-time\nleakage audit\n\nT48: 4 events,\nunestimable", "#fbe9e7", RED),
    (60, "Admission model\n\nOOF AUROC 0.735\nLow/Int/High\n15 / 32 / 73%", "#e8f0f7", BLUE),
    (79, "Governance\n\nrecalibration\nabstention\neICU audit", GREY, NAVY),
]

w, h, y0 = 16, 22, 8
for x, txt, fc, ec in stages:
    ax.add_patch(
        FancyBboxPatch(
            (x, y0),
            w,
            h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=1.3,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(x + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=7.2, color="#1a1a1a")

for x in (19, 38, 57, 76):
    ax.add_patch(
        FancyArrowPatch(
            (x, y0 + h / 2),
            (x + 3.2, y0 + h / 2),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.4,
            color=NAVY,
        )
    )

ax.text(
    50,
    4.6,
    "Front-loaded early escalation makes dynamic 48 h re-prediction unestimable; "
    "the defensible output is admission surveillance-priority stratification (not treatment).",
    ha="center",
    va="center",
    fontsize=6.8,
    color="#333333",
    wrap=True,
)

fig.savefig(os.path.join(FIG, "graphical_abstract.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "graphical_abstract.png"), dpi=600, bbox_inches="tight")
plt.close(fig)
print("-> graphical_abstract")
