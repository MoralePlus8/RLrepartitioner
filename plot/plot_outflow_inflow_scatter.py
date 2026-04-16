#!/usr/bin/env python3
"""
从 stats/srrip_2core 目录下所有 CSV 文件读取数据，
从第二次 cpu0 和 cpu1 的 heartbeat 均为 0 开始，
以 period_eviction_count 为 outflow (x)，period_fill_count 为 inflow (y) 绘制散点图。
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path

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

x_points = []  # outflow (period_eviction_count)
y_points = []  # inflow  (period_fill_count)

stats_dir = Path('stats/srrip_2core')
csv_files = sorted(stats_dir.glob('*.csv'))

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"跳过 {csv_path.name}: 读取失败 - {e}")
        continue

    hb_zero_indices = df.index[df['heartbeat'] == 0].tolist()
    if len(hb_zero_indices) < 3:
        print(f"跳过 {csv_path.name}: heartbeat=0 次数不足")
        continue

    second_group_start = None
    for k in range(1, len(hb_zero_indices)):
        if hb_zero_indices[k] - hb_zero_indices[k - 1] > 1:
            second_group_start = hb_zero_indices[k]
            break

    if second_group_start is None:
        print(f"跳过 {csv_path.name}: 未找到第二次 heartbeat=0")
        continue

    df_valid = df.iloc[second_group_start:]
    x_points.extend(df_valid['period_eviction_count'].values)
    y_points.extend(df_valid['period_fill_count'].values)

print(f"共读取 {len(csv_files)} 个 CSV 文件")

x_points = np.array(x_points, dtype=float)
y_points = np.array(y_points, dtype=float)
n_total = len(x_points)

if n_total == 0:
    print("无有效数据，退出")
    exit(1)

outlier_mask = (x_points <= 8 * y_points) & (y_points <= 8 * x_points)
x_points = x_points[outlier_mask]
y_points = y_points[outlier_mask]
n_after_outlier = len(x_points)

correlation_matrix = np.corrcoef(x_points, y_points)
r = correlation_matrix[0, 1]
r_squared = r ** 2
valid = y_points != 0
mape = np.mean(np.abs((y_points[valid] - x_points[valid]) / y_points[valid])) * 100 if np.any(valid) else np.nan

positive_mask = (x_points > 0) & (y_points > 0)
x_plot = x_points[positive_mask]
y_plot = y_points[positive_mask]
print(f"过滤零值后剩余 {len(x_plot)} 个点（移除 {(~positive_mask).sum()} 个零值点）")

range_mask = (x_plot >= 100) & (y_plot >= 100)
x_plot = x_plot[range_mask]
y_plot = y_plot[range_mask]
print(f"过滤 <10^2 后剩余 {len(x_plot)} 个点（移除 {(~range_mask).sum()} 个）")

print("正在计算 KDE 密度（在 log 空间上）...")
log_x = np.log10(x_plot)
log_y = np.log10(y_plot)
xy_stack = np.vstack([log_x, log_y])
n_pts = xy_stack.shape[1]
KDE_SAMPLE_LIMIT = 20000
if n_pts > KDE_SAMPLE_LIMIT:
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(n_pts, KDE_SAMPLE_LIMIT, replace=False)
    kde = gaussian_kde(xy_stack[:, idx_sample])
else:
    kde = gaussian_kde(xy_stack)
density = kde(xy_stack)
sort_idx = np.argsort(density)
x_sorted = x_plot[sort_idx]
y_sorted = y_plot[sort_idx]
density_sorted = density[sort_idx]

fig, ax = plt.subplots(figsize=(5.5, 3.0))

COLOR_PERFECT = '#dc2626'

max_val = max(x_plot.max(), y_plot.max())
line_range = np.logspace(2, np.log10(max_val), 200)

ax.plot(line_range, line_range, color=COLOR_PERFECT, linewidth=0.8,
        linestyle='--', alpha=0.7, zorder=4, label='y = x (perfect)')

sc = ax.scatter(x_sorted, y_sorted, c=density_sorted, cmap='viridis',
                alpha=0.6, s=3, edgecolors='none', zorder=2, rasterized=True)

cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
cbar.set_label('Density', fontsize=9)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(100, max_val * 1.1)
ax.set_ylim(100, max_val * 1.1)

ax.set_xlabel('Outflow (evictions)', fontweight='medium')
ax.set_ylabel('Inflow (misses)', fontweight='medium')

ax.grid(True, which='major', linestyle='-', alpha=0.25, color='#555555')
ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='#888888')

ax.legend(loc='lower right', framealpha=0.95, edgecolor='#cccccc')

textstr = (r'$r^2$ = ' + f'{r_squared:.3f}' + '\n' +
          r'MAPE = ' + f'{mape:.1f}%')
props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc',
             alpha=0.95, linewidth=1)
ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=8.5,
        verticalalignment='top', bbox=props, family='monospace')

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')

plt.tight_layout()
plt.savefig('plot/outflow_inflow_scatter.png', dpi=300, facecolor='white')
plt.savefig('plot/outflow_inflow_scatter.pdf', facecolor='white')
plt.close()

print(f"去除离群点后共 {n_after_outlier} 个点（原 {n_total} 个，剔除 {n_total - n_after_outlier} 个）")
print(f"皮尔逊相关系数 r = {r:.4f}")
print(f"决定系数 r² = {r_squared:.4f}")
print(f"平均绝对百分比误差 MAPE = {mape:.2f}%")
print(f"图像已保存为 outflow_inflow_scatter.png (300 DPI) 和 outflow_inflow_scatter.pdf")
