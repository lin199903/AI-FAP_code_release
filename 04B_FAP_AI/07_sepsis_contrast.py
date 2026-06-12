"""
Negative-control contrast for the event-timing audit.

Claim being tested: the audit does NOT reflexively reject dynamic 24/48 h modelling --
it greenlights it when the disease's event clock is genuinely day-scale. We therefore run
the same event-timing / at-risk-by-landmark step on a classic day-scale phenotype: in-hospital
mortality among ICU patients with sepsis (t0 = ICU admission). If events are spread over days
and 24/48 h landmarks retain many at-risk events (unlike MDAP's 5/4), the audit correctly
preserves dynamic modelling here.

Outputs:
  outputs/sepsis_contrast_timing.csv      (cumulative event timing + at-risk events by landmark)
  manuscript/JBI/figures/figureS3_contrast.{pdf,png}
"""
import os, psycopg2, psycopg2.extras, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _dbconfig import get_dsn
DSN = get_dsn()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FIG_DIRS = [os.path.normpath(os.path.join(OUT, "..", "..", "manuscript", "JBI", "figures")),
            os.path.normpath(os.path.join(OUT, "..", "..", "manuscript", "latex", "figures"))]
conn = psycopg2.connect(DSN); conn.autocommit = True
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

SQL = """
WITH sepsis_hadm AS (
  SELECT DISTINCT d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version = 9  AND d.icd_code IN ('99591','99592','78552'))
     OR (d.icd_version = 10 AND (LEFT(d.icd_code,3) IN ('A40','A41') OR d.icd_code IN ('R6520','R6521')))
),
first_icu AS (
  SELECT DISTINCT ON (icu.hadm_id) icu.hadm_id, icu.subject_id, icu.stay_id,
         icu.intime AS t0, icu.outtime, icu.los AS icu_los_days
  FROM mimiciv_icu.icustays icu
  JOIN sepsis_hadm s ON icu.hadm_id = s.hadm_id
  ORDER BY icu.hadm_id, icu.intime
),
cohort AS (
  SELECT f.hadm_id, f.subject_id, f.t0, f.icu_los_days,
         a.deathtime, a.hospital_expire_flag, a.dischtime,
         p.anchor_age AS age
  FROM first_icu f
  JOIN mimiciv_hosp.admissions a ON f.hadm_id = a.hadm_id
  JOIN mimiciv_hosp.patients   p ON f.subject_id = p.subject_id
  WHERE p.anchor_age >= 18
)
SELECT hadm_id, age, hospital_expire_flag, icu_los_days,
       EXTRACT(EPOCH FROM (deathtime - t0))/3600.0 AS death_offset_h,
       EXTRACT(EPOCH FROM (dischtime  - t0))/3600.0 AS disch_offset_h
FROM cohort;
"""
cur.execute(SQL)
df = pd.DataFrame(cur.fetchall())
conn.close()
df["death_offset_h"] = pd.to_numeric(df["death_offset_h"], errors="coerce")
df["disch_offset_h"] = pd.to_numeric(df["disch_offset_h"], errors="coerce")
df["icu_los_h"] = pd.to_numeric(df["icu_los_days"], errors="coerce") * 24.0
N = len(df)
died = (df["hospital_expire_flag"] == 1) & df["death_offset_h"].notna() & (df["death_offset_h"] > 0)
ev = df.loc[died, "death_offset_h"].values
ev = ev[(ev > 0) & (ev <= 24 * 60)]  # within 60 days, post-ICU
nE = len(ev)
print(f"Sepsis-ICU cohort: N={N}  in-hospital deaths (post-ICU, <=60d)={nE} ({100*nE/N:.1f}%)")

def pct_by(h): return 100 * (ev <= h).mean()
timing = {h: pct_by(h) for h in [6, 24, 48, 72, 168]}
print("  cumulative deaths by:", {f"{h}h": f"{p:.1f}%" for h, p in timing.items()})

# at-risk + future deaths by landmark (parallel to MDAP Figure 3b)
rows = []
for L in [0, 24, 48, 72]:
    if L == 0:
        at_risk = np.ones(N, bool)
    else:
        d = df["death_offset_h"]; o = df["disch_offset_h"]; los = df["icu_los_h"]
        at_risk = ((d.isna() | (d > L)) & (los > L)).values
    dh = df["death_offset_h"].values
    fut = died.values & (dh > L) & (dh <= 24 * 60)
    rows.append(dict(landmark=f"T{L}", at_risk_N=int(at_risk.sum()),
                     future_deaths=int((fut & at_risk).sum())))
contrast = pd.DataFrame(rows)
print(contrast.to_string(index=False))
contrast.to_csv(os.path.join(OUT, "sepsis_contrast_timing.csv"), index=False)

# figure
NAVY, RED, GREY, GREEN = "#22405f", "#c0392b", "#9aa3ab", "#2e7d52"
MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 70 * MM))
gs = GridSpec(1, 2, figure=fig, wspace=0.34, width_ratios=[1.25, 1])
ax = fig.add_subplot(gs[0, 0])
evs = np.sort(ev); cum = np.arange(1, nE + 1) / nE * 100
ax.step(np.concatenate([[0], evs]), np.concatenate([[0], cum]), where="post", color=NAVY, lw=1.6, label="Sepsis ICU mortality")
# overlay MDAP front-loading for contrast
mdap = {6: 66.4, 24: 81.9, 48: 88.6}
for h, lab in [(24, "24 h"), (48, "48 h")]:
    ax.axvline(h, color=GREY, lw=0.7, ls="--")
    ax.plot(h, pct_by(h), "o", color=RED, ms=4, zorder=5)
    ax.annotate(f"{pct_by(h):.0f}% by {lab}\n(MDAP {mdap[h]:.0f}%)", (h, pct_by(h)),
                textcoords="offset points", xytext=(6, -4), fontsize=5.6, color=RED)
ax.set_xlim(0, 168); ax.set_ylim(0, 102)
ax.set_xlabel("Hours after ICU admission"); ax.set_ylabel("Cumulative deaths (%)")
ax.set_title("Day-scale clock: sepsis mortality is NOT front-loaded", fontsize=7.4)
ax.text(-0.16, 1.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(contrast))
ax.bar(x, contrast["future_deaths"], color=[GREEN, NAVY, NAVY, GREY], edgecolor="black", lw=0.5, width=0.66)
for i, v in enumerate(contrast["future_deaths"]):
    ax.text(i, v + max(contrast["future_deaths"]) * 0.02, str(v), ha="center", fontsize=6.2)
ax.set_xticks(x); ax.set_xticklabels(contrast["landmark"])
ax.set_ylabel("At-risk future deaths (n)")
ax.set_title("24/48 h landmarks retain ample events\n(contrast: MDAP retained 5 / 4)", fontsize=7.4)
ax.text(-0.16, 1.06, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
for d in FIG_DIRS:
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "figureS3_contrast.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(d, "figureS3_contrast.png"), dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved figureS3_contrast to", "; ".join(FIG_DIRS))
