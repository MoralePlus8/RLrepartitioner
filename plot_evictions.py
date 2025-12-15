#!/usr/bin/env python3
"""
从 llc_stats.csv 读取数据，计算预测驱逐值并与实际值对比绘图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv('llc_stats.csv')

# 存储所有点的坐标
x_points = []  # 预测值 (E_hat)
y_points = []  # 实际值 (E)

# 每两行作为一组进行处理
for i in range(0, len(df) - 1, 2):
    row1 = df.iloc[i]  # 第一行
    row2 = df.iloc[i + 1]  # 第二行
    
    # 提取属性
    L1 = row1['little_law_lifetime']
    L2 = row2['little_law_lifetime']
    W1 = row1['period_avg_way_occupancy']
    W2 = row2['period_avg_way_occupancy']
    E1 = row1['period_total_evictions_caused']
    E2 = row2['period_total_evictions_caused']
    E2_1 = row1['period_evictions_caused']  # 第一行的 period_evictions_caused
    E1_2 = row2['period_evictions_caused']  # 第二行的 period_evictions_caused
    
    # 计算分母
    denominator = L1 * W2 + L2 * W1
    
    # 避免除以零
    if denominator == 0:
        continue
    
    # 计算预测值
    E2_1_hat = E1 * (L1 * W2 / denominator)  # 预测的 E2-1
    E1_2_hat = E2 * (L2 * W1 / denominator)  # 预测的 E1-2
    
    # 添加点 (E2_1_hat, E2_1) 和 (E1_2_hat, E1_2)
    x_points.append(E2_1_hat)
    y_points.append(E2_1)
    x_points.append(E1_2_hat)
    y_points.append(E1_2)

# 转换为numpy数组
x_points = np.array(x_points)
y_points = np.array(y_points)

# 计算皮尔逊相关系数
correlation_matrix = np.corrcoef(x_points, y_points)
r = correlation_matrix[0, 1]
r_squared = r ** 2

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

# 在图上显示相关系数
textstr = f'r = {r:.4f}\n$r^2$ = {r_squared:.4f}'
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

print(f"共绘制了 {len(x_points)} 个点")
print(f"皮尔逊相关系数 r = {r:.4f}")
print(f"决定系数 r² = {r_squared:.4f}")
print(f"图像已保存为 evictions_comparison.png")

