# -*- coding: utf-8 -*-
"""
AI-FAP 旗舰 — 六类监测风险分型映射（Action Mapping）
===============================================================================
按 AGENTS.md §6.3 执行，将治理层输出转化为临床监测优先级分型。

逻辑：
  1. Abstention: 治理层识别的低置信度样本 (L3 uncertainty)
  2. Gray-zone: 中等风险 (0.35 <= p < 0.65)
  3. Routine: 低风险 (p < 0.35) 且确定性高
  4. Vulnerable: 高风险 (p >= 0.65) + 轨迹表型细分
     - Inflammatory-vulnerable: 高风险 + GBTM 炎症/肾脏恶化轨迹
     - Metabolic-vulnerable: 高风险 + 入院高 TG 且 轨迹未恶化
     - Dual-vulnerable: 高风险 + 入院高 TG + 轨迹恶化

输入：04B_FAP_AI/outputs/
  - landmark_ml_predictions.csv
  - canonical_mdap_cohort.csv
输出：04B_FAP_AI/outputs/
  - risk_typing_results.csv
  - risk_typing_summary.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 1. 加载数据
df_pred = pd.read_csv(os.path.join(OUT, "landmark_ml_predictions.csv"))
df_cohort = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))

# 合并 tg_ge500_flag (df_pred 已含 gbtm_class, composite_outcome)
df = df_pred.merge(df_cohort[['hadm_id', 'tg_ge500_flag']], on='hadm_id', how='left')

# 2. 分型算法
# 定义不确定性 (Abstention) 阈值: 选取置信度最低的 20%
# Note: this six-type action map intentionally uses a conservative gray-zone band
# (0.35-0.65) after the uncertainty gate. The separate 0.20/0.50 cut-points used
# elsewhere in governance scripts are for low/intermediate/high probability summaries,
# not for the final six-type bedside mapping reported in the manuscript.
df['confidence'] = (df['prob_lgb_T0'] - 0.5).abs()
abstention_threshold = df['confidence'].quantile(0.20)

def map_risk_type(row):
    p = row['prob_lgb_T0']
    conf = row['confidence']
    traj = row['gbtm_class'] # 0: Inflammatory, 1: Rapid, 2: Renal
    tg_high = row['tg_ge500_flag']
    
    # Tier 1: Abstention (Uncertainty)
    if conf < abstention_threshold:
        return 'Abstention'
    
    # Tier 2: Gray-zone (Indeterminate)
    if 0.35 <= p < 0.65:
        return 'Gray-zone'
    
    # Tier 3: Routine (Low risk)
    if p < 0.35:
        return 'Routine-monitoring'
    
    # Tier 4: Vulnerable (High risk)
    if p >= 0.65:
        if (traj in [0, 2]) and (tg_high == 1):
            return 'Dual-vulnerable'
        elif (traj in [0, 2]):
            return 'Inflammatory-vulnerable'
        else:
            return 'Metabolic-vulnerable'
    
    return 'Unknown'

df['risk_type'] = df.apply(map_risk_type, axis=1)

# 3. 统计汇总
summary = df.groupby('risk_type').agg(
    n=('hadm_id', 'count'),
    outcome_rate=('composite_outcome', 'mean'),
    avg_prob=('prob_lgb_T0', 'mean')
).reset_index()

summary['outcome_rate'] = (summary['outcome_rate'] * 100).round(1)
summary['avg_prob'] = summary['avg_prob'].round(3)

print("\nAI-FAP Risk Typing Summary:")
print(summary.to_string(index=False))

# 4. 保存结果
df.to_csv(os.path.join(OUT, "risk_typing_results.csv"), index=False)

# 5. 可视化
plt.figure(figsize=(10, 6))
colors = {
    'Routine-monitoring': '#4CAF50',
    'Gray-zone': '#9E9E9E',
    'Abstention': '#607D8B',
    'Metabolic-vulnerable': '#FF9800',
    'Inflammatory-vulnerable': '#F44336',
    'Dual-vulnerable': '#B71C1C'
}

# 排序以匹配风险梯度
type_order = ['Routine-monitoring', 'Abstention', 'Gray-zone', 'Metabolic-vulnerable', 'Inflammatory-vulnerable', 'Dual-vulnerable']
summary_sorted = summary.set_index('risk_type').reindex(type_order).reset_index().dropna()

bars = plt.bar(summary_sorted['risk_type'], summary_sorted['outcome_rate'], color=[colors.get(t, '#000') for t in summary_sorted['risk_type']])
plt.xticks(rotation=30, ha='right')
plt.ylabel('Outcome Rate (%)')
plt.title('AI-FAP: Actionable Risk Typing and Surveillance Priority')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "risk_typing_summary.png"), dpi=200)
print(f"\nSummary figure saved to {os.path.join(OUT, 'risk_typing_summary.png')}")
