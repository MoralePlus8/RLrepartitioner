#!/usr/bin/env python3
"""
从 stats/srrip_2core 目录下所有 CSV 文件读取数据，计算预测驱逐值并与实际值对比绘图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# 去除离群点：预测值>5倍实际值 或 实际值>5倍预测值
outlier_mask = (x_points <= 8 * y_points) & (y_points <= 8 * x_points)
x_points = x_points[outlier_mask]
y_points = y_points[outlier_mask]

# 计算皮尔逊相关系数
correlation_matrix = np.corrcoef(x_points, y_points)
r = correlation_matrix[0, 1]
r_squared = r ** 2

# 计算平均绝对百分比误差 MAPE = mean(|actual - predicted| / actual) * 100
# 排除 actual=0 的点避免除零
valid = y_points != 0
mape = np.mean(np.abs((y_points[valid] - x_points[valid]) / y_points[valid])) * 100 if np.any(valid) else np.nan

# 创建图形
plt.figure(figsize=(10, 10))

# 绘制散点图
plt.scatter(x_points, y_points, alpha=0.5, s=20, c='steelblue', label='Data Points')

# 使用对数坐标轴让数据点分布更均匀
plt.xscale('log')
plt.yscale('log')

# 绘制 y=x 直线（对数坐标下）
max_val = max(x_points.max(), y_points.max())
min_val = max(x_points.min(), y_points.min(), 1)  # 避免对数坐标下的0值问题
line_range = np.logspace(np.log10(min_val), np.log10(max_val), 100)
plt.plot(line_range, line_range, 'r-', linewidth=2, label='y = x')

# 设置标签和标题
plt.xlabel(r'Predicted Evictions ($\hat{E}$)', fontsize=12)
plt.ylabel(r'Actual Evictions (E)', fontsize=12)
plt.title('Predicted vs Actual Evictions Caused (Log Scale)', fontsize=14)
plt.legend(fontsize=10, loc='lower right')

# 在图上显示相关系数和 MAPE
textstr = f'r = {r:.4f}\n$r^2$ = {r_squared:.4f}\nMAPE = {mape:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

# 设置等比例坐标轴（对数坐标下）
plt.axis('equal')
plt.grid(True, alpha=0.3, which='both')

# 保存图像
plt.tight_layout()
plt.savefig('evictions_comparison.png', dpi=150)
plt.show()

print(f"去除离群点后共 {len(x_points)} 个点（原 {n_total} 个，剔除 {n_total - len(x_points)} 个）")
print(f"皮尔逊相关系数 r = {r:.4f}")
print(f"决定系数 r² = {r_squared:.4f}")
print(f"平均绝对百分比误差 MAPE = {mape:.2f}%")
print(f"图像已保存为 evictions_comparison.png")

