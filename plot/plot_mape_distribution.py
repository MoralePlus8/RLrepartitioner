#!/usr/bin/env python3
"""
从 stats/srrip_2core 目录下所有 CSV 文件读取数据，
每个文件单独计算 MAPE，绘制频率分布柱状图。
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============ 论文插图风格配置（适配 Word 5号字 = 10.5pt）============
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10.5,
    'axes.titlesize': 10.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10.5,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
})

stats_dir = Path('stats/srrip_2core')
csv_files = sorted(stats_dir.glob('*.csv'))

mape_values = []
miss_rates = []
mape_files = []

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"跳过 {csv_path.name}: 读取失败 - {e}")
        continue

    actuals = []
    preds = []

    for i in range(0, len(df) - 1, 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        E1 = row1['period_misses']
        E2 = row2['period_misses']
        E2_1 = row1['period_evictions_caused_without_wb']
        E1_2 = row2['period_evictions_caused_without_wb']

        denominator = E1 + E2
        if denominator == 0:
            continue

        E2_1_hat = E1 * E2 / denominator
        E1_2_hat = E2 * E1 / denominator

        if E2_1 != 0 and E2_1_hat <= E2_1 * 8 and E2_1 <= E2_1_hat * 8:
            actuals.append(E2_1)
            preds.append(E2_1_hat)
        if E1_2 != 0 and E1_2_hat <= E1_2 * 8 and E1_2 <= E1_2_hat * 8:
            actuals.append(E1_2)
            preds.append(E1_2_hat)

    if len(actuals) == 0:
        continue

    actuals = np.array(actuals)
    preds = np.array(preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    mape_values.append(mape)
    mape_files.append(csv_path.stem)

    per_cpu = df.groupby('cpu').agg(
        total_misses=('period_misses', 'sum'),
        total_accesses=('period_accesses', 'sum'),
    )
    per_cpu['miss_rate'] = per_cpu['total_misses'] / per_cpu['total_accesses']
    miss_rates.append(per_cpu['miss_rate'].mean())

mape_values = np.array(mape_values)
miss_rates = np.array(miss_rates)
print(f"共读取 {len(csv_files)} 个 CSV 文件，有效 MAPE 值 {len(mape_values)} 个")
print(f"MAPE 范围: {mape_values.min():.2f}% ~ {mape_values.max():.2f}%")
print(f"MAPE 均值: {mape_values.mean():.2f}%, 中位数: {np.median(mape_values):.2f}%")

for pct in [50, 75, 90, 95, 99]:
    print(f"  P{pct}: {np.percentile(mape_values, pct):.2f}%")

high_mape = [(f, m) for f, m in zip(mape_files, mape_values) if m >= 100]
high_mape.sort(key=lambda x: x[1], reverse=True)
print(f"\n===== MAPE >= 100% 的工作负载组合 ({len(high_mape)} 个) =====")
for rank, (name, m) in enumerate(high_mape, 1):
    print(f"  {rank:3d}. {m:10.2f}%  {name}")

# ============ 频率分布柱状图 ============
bin_width = 2
cap = 100  # 超过此值归入最后一个 ">cap%" 区间
bins = np.arange(0, cap + bin_width, bin_width)

mape_normal = mape_values[mape_values < cap]
n_overflow = int(np.sum(mape_values >= cap))

fig, ax = plt.subplots(figsize=(5.5, 3.0))

COLOR_BAR = '#2563eb'

counts, edges, _ = ax.hist(
    mape_normal, bins=bins,
    color=COLOR_BAR, edgecolor='white', linewidth=0.8,
    alpha=0.85, zorder=2,
)

if n_overflow > 0:
    overflow_left = cap
    overflow_right = cap + bin_width
    ax.bar((overflow_left + overflow_right) / 2, n_overflow, width=bin_width,
           color='#dc2626', edgecolor='white', linewidth=0.8,
           alpha=0.85, zorder=2)

tick_positions = np.arange(0, cap + bin_width * 2, 10)
tick_labels = [f'{int(t)}' for t in tick_positions]
tick_labels[-1] = f'>{cap}'
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

ax.set_xlabel('MAPE (%)', fontweight='medium')
ax.set_ylabel('Number of Workloads', fontweight='medium')

ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.25, color='#555555')

# ============ 右侧纵轴：每个细分区间的平均 miss rate ============
COLOR_LINE = '#e67e22'

all_bins = np.append(bins, cap + bin_width)
bin_indices = np.digitize(mape_values, bins) - 1
bin_indices = np.clip(bin_indices, 0, len(all_bins) - 2)

avg_miss_rate_per_bin = []
bin_centers = []
for b in range(len(all_bins) - 1):
    mask = bin_indices == b
    if mask.sum() > 0:
        avg_mr = miss_rates[mask].mean() * 100
        avg_miss_rate_per_bin.append(avg_mr)
        bin_centers.append((all_bins[b] + all_bins[b + 1]) / 2)

ax2 = ax.twinx()
ax2.plot(bin_centers, avg_miss_rate_per_bin, color=COLOR_LINE, linewidth=1.0,
         marker='o', markersize=2.5, zorder=5, label='Avg. Miss Rate')
ax2.set_ylabel('Avg. Cache Miss Rate (%)', fontweight='medium', color=COLOR_LINE, fontsize=10.5)
ax2.tick_params(axis='y', labelcolor=COLOR_LINE)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')

# ============ 图例（含 Median MAPE 信息）============
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=COLOR_BAR, alpha=0.85, edgecolor='white', label='MAPE < 100%'),
]
if n_overflow > 0:
    legend_elements.append(
        Patch(facecolor='#dc2626', alpha=0.85, edgecolor='white',
              label=f'MAPE $\\geq$ 100% ({n_overflow})'),
    )
legend_elements.append(
    Line2D([0], [0], color=COLOR_LINE, linewidth=1.0, marker='o', markersize=2.5,
           label='Avg. Miss Rate'),
)

ax.legend(handles=legend_elements, loc='upper right',
          framealpha=0.95, edgecolor='#cccccc')

plt.tight_layout()
plt.savefig('plot/mape_distribution.png', dpi=300, facecolor='white')
plt.savefig('plot/mape_distribution.pdf', facecolor='white')
plt.close()

print(f"图像已保存为 plot/mape_distribution.png (300 DPI) 和 plot/mape_distribution.pdf")
