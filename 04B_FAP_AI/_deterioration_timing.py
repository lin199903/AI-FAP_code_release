# -*- coding: utf-8 -*-
"""
R1 + 结局效度质控：MDAP 临床恶化的时间分布。
回答两个问题：
  (1) 复合恶化事件（ICU 7d OR 院内死亡）在入院后何时发生 → 是否"前置"？
  (2) ICU 事件是否大量贴在 T0（直接/急诊入 ICU）→ 流程性偏倚 vs 真实进展？
输入: outputs/canonical_mdap_cohort.csv
"""
import os, pandas as pd, numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))
N = len(df)
print(f"MDAP cohort N={N}")

icu_h = df["icu_intime_hours"]
dth_h = df["death_offset_hours"]
df["composite"] = ((df["icu_7d"] == 1) | (df["hospital_expire_flag"] == 1)).astype(int)
n_evt = int(df["composite"].sum())
print(f"Composite events (ICU7d OR hosp death): {n_evt} ({100*n_evt/N:.1f}%)")

# ---- (1) ICU intime 相对 T0 的分布（含 <=0 = 直接/转入 ICU）----
print("\n[1] ICU intime relative to admission (all with an ICU stay):")
icu_any = icu_h.dropna()
bins = [(-1e9, 0, "<=0h (direct/transfer ICU)"), (0, 6, "0-6h"), (6, 24, "6-24h"),
        (24, 48, "24-48h"), (48, 168, "48-168h"), (168, 1e9, ">168h")]
for lo, hi, lab in bins:
    n = int(((icu_any > lo) & (icu_any <= hi)).sum()) if lo != -1e9 else int((icu_any <= hi).sum())
    print(f"  {lab:28s}: {n:4d} ({100*n/max(len(icu_any),1):.1f}% of ICU)")
print(f"  total with ICU stay: {len(icu_any)}")

# ---- (2) 复合事件首发时间（ICU7d 用 icu_intime_hours，死亡用 death_offset_hours）----
print("\n[2] Time of FIRST composite event (front-loading):")
def event_time(r):
    times = []
    if r["icu_7d"] == 1 and pd.notna(r["icu_intime_hours"]) and r["icu_intime_hours"] > 0:
        times.append(r["icu_intime_hours"])
    if r["hospital_expire_flag"] == 1 and pd.notna(r["death_offset_hours"]) and r["death_offset_hours"] > 0:
        times.append(r["death_offset_hours"])
    return min(times) if times else np.nan

df["evt_time_h"] = df.apply(event_time, axis=1)
evt = df.loc[df["composite"] == 1, "evt_time_h"].dropna()
cum_bins = [(0, 6), (6, 24), (24, 48), (48, 168), (168, 1e9)]
cum = 0
for lo, hi in cum_bins:
    n = int(((evt > lo) & (evt <= hi)).sum())
    cum += n
    lab = f">{int(lo)}h" if hi > 1e8 else f"{int(lo)}-{int(hi)}h"
    print(f"  {lab:10s}: {n:4d} events  | cumulative {cum}/{len(evt)} ({100*cum/max(len(evt),1):.1f}%)")
le24 = int((evt <= 24).sum()); le48 = int((evt <= 48).sum())
print(f"\n  >>> within 24h: {le24}/{len(evt)} ({100*le24/len(evt):.1f}%)")
print(f"  >>> within 48h: {le48}/{len(evt)} ({100*le48/len(evt):.1f}%)")
print(f"  >>> after 48h:  {len(evt)-le48}/{len(evt)} ({100*(len(evt)-le48)/len(evt):.1f}%)")

# ---- (3) 结局效度：ICU 在 <=6h 的占比（high-acuity disposition 信号）----
n_icu_total = int((df["icu_7d"] == 1).sum())
icu7 = df.loc[df["icu_7d"] == 1, "icu_intime_hours"].dropna()
n_early = int((icu7 <= 6).sum())
print(f"\n[3] Outcome-validity check:")
print(f"  ICU within 7d: {n_icu_total}; of these intime<=6h: {n_early} ({100*n_early/max(n_icu_total,1):.1f}%)")
print(f"  -> 若该比例很高，ICU 结局更接近 'early escalation / high-acuity disposition' 而非后期进展")

df[["hadm_id", "composite", "evt_time_h", "icu_intime_hours", "icu_7d",
    "death_offset_hours", "hospital_expire_flag", "los_days"]].to_csv(
    os.path.join(OUT, "deterioration_timing.csv"), index=False)
print(f"\n-> {os.path.join(OUT, 'deterioration_timing.csv')}")
