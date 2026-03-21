#!/usr/bin/env python3
"""
从 stats/srrip_2core 目录下所有 CSV 文件读取数据，计算预测驱逐值并与实际值对比绘图
论文级插图：散点图（横轴真实值，纵轴预测值，低透明度+小尺寸展示全部数据点密度）
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============ 论文插图风格配置 ============
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#333333',
})

# 存储所有点的坐标
x_points = []  # 预测值 (E_hat)
y_points = []  # 实际值 (E)

stats_dir = Path('stats/srrip_2core')
csv_files = sorted(stats_dir.glob('*.csv'))

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"跳过 {csv_path.name}: 读取失败 - {e}")
        continue

    # 每两行作为一组进行处理
    for i in range(0, len(df) - 1, 2):
        row1 = df.iloc[i]  # 第一行
        row2 = df.iloc[i + 1]  # 第二行

        E1 = row1['period_misses']
        E2 = row2['period_misses']
        E2_1 = row1['period_evictions_caused_without_wb']
        E1_2 = row2['period_evictions_caused_without_wb']

        denominator = E1 + E2
        if denominator == 0:
            continue

        E2_1_hat = E1 * E2 / denominator
        E1_2_hat = E2 * E1 / denominator

        x_points.append(E2_1_hat)
        y_points.append(E2_1)
        x_points.append(E1_2_hat)
        y_points.append(E1_2)

print(f"共读取 {len(csv_files)} 个 CSV 文件")

# 转换为numpy数组
x_points = np.array(x_points)
y_points = np.array(y_points)
n_total = len(x_points)

if n_total == 0:
    print("无有效数据，退出")
    exit(1)

# 去除离群点：预测值>8倍实际值 或 实际值>8倍预测值
outlier_mask = (x_points <= 8 * y_points) & (y_points <= 8 * x_points)
x_points = x_points[outlier_mask]
y_points = y_points[outlier_mask]
n_after_outlier = len(x_points)

# 在完整数据上计算统计量
correlation_matrix = np.corrcoef(x_points, y_points)
r = correlation_matrix[0, 1]
r_squared = r ** 2
valid = y_points != 0
mape = np.mean(np.abs((y_points[valid] - x_points[valid]) / y_points[valid])) * 100 if np.any(valid) else np.nan

# 横轴=真实值(E)，纵轴=预测值(E_hat)
x_actual = y_points   # 真实观测值
y_pred = x_points     # 模型预测值

# 过滤掉零值点（对数坐标下无法显示）
positive_mask = (x_actual > 0) & (y_pred > 0)
x_actual = x_actual[positive_mask]
y_pred = y_pred[positive_mask]
print(f"过滤零值后剩余 {len(x_actual)} 个点（移除 {(~positive_mask).sum()} 个零值点）")

# ============ 创建论文级插图（横轴长度为纵轴两倍）============
fig, ax = plt.subplots(figsize=(12, 6))

COLOR_SCATTER = '#2563eb'
COLOR_PERFECT = '#dc2626'

# 计算坐标范围
min_val = max(x_actual.min(), y_pred.min(), 1)
max_val = max(x_actual.max(), y_pred.max())
line_range = np.logspace(np.log10(min_val), np.log10(max_val), 200)

# 绘制 y=x 理想线
ax.plot(line_range, line_range, color=COLOR_PERFECT, linewidth=2,
        linestyle='-.', zorder=3, label='y = x (perfect)')

# 散点图：低透明度 + 小尺寸，通过重叠自然呈现密度
ax.scatter(x_actual, y_pred, c=COLOR_SCATTER, alpha=0.08, s=4,
           edgecolors='none', zorder=2, rasterized=True)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(10, max_val * 1.1)
ax.set_ylim(10, max_val * 1.1)

ax.set_xlabel(r'Actual Evictions ($E$)', fontweight='medium')
ax.set_ylabel(r'Predicted Evictions ($\hat{E}$)', fontweight='medium')
ax.set_title('Prediction Accuracy (All Points)', fontsize=13, fontweight='medium')

ax.grid(True, which='major', linestyle='-', alpha=0.25, color='#555555')
ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='#888888')

ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc')

textstr = (r'$r^2$ = ' + f'{r_squared:.3f}' + '\n' +
          r'MAPE = ' + f'{mape:.1f}%')
props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc',
             alpha=0.95, linewidth=1)
ax.text(0.04, 0.04, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=props, family='monospace')

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')

plt.tight_layout()
plt.savefig('evictions_comparison.png', dpi=300, facecolor='white')
plt.savefig('evictions_comparison.pdf', facecolor='white')
plt.close()

print(f"去除离群点后共 {n_after_outlier} 个点（原 {n_total} 个，剔除 {n_total - n_after_outlier} 个）")
print(f"皮尔逊相关系数 r = {r:.4f}")
print(f"决定系数 r² = {r_squared:.4f}")
print(f"平均绝对百分比误差 MAPE = {mape:.2f}%")
print(f"图像已保存为 evictions_comparison.png (300 DPI) 和 evictions_comparison.pdf")

