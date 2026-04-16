#!/usr/bin/env python3
"""
绘制 4 核心配置下 Even / UCP / RL 三种策略相对于 LRU 基准的归一化吞吐量 S 曲线。
所有策略按 UCP 的排序顺序排列。
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

# ============ 目录配置 ============
BASE_DIR = Path('stats')
LRU_DIR = BASE_DIR / 'lru_4core'
TRAIN_DIR = BASE_DIR / 'train'
STRATEGIES = {
    'UCP':  BASE_DIR / 'ucp_4core',
    'Even Part': BASE_DIR / 'even_4core',
    'RL':   BASE_DIR / 'rl_4core_v2',
}


def compute_ipc(csv_path: Path) -> float | None:
    """从 CSV 文件中读取最后 4 行（4 个核心），计算总体 IPC。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if len(df) < 4:
        return None
    last_four = df.iloc[-4:]
    total_instructions = last_four['instructions'].sum()
    global_cycle = last_four['global_cycle'].iloc[0]
    if global_cycle == 0:
        return None
    return total_instructions / global_cycle


def resolve_strategy_path(strategy_dir: Path, trace_name: str) -> Path | None:
    """根据策略目录和 trace 名找到对应的 CSV 文件路径。"""
    direct = strategy_dir / trace_name
    if direct.exists():
        return direct
    prefixed = strategy_dir / f"rl_{trace_name}"
    if prefixed.exists():
        return prefixed
    return None


# ============ 构建训练集排除列表 ============
train_traces = set()
for f in TRAIN_DIR.glob('*.csv'):
    name = f.name
    if name.startswith('rl_'):
        name = name[3:]
    train_traces.add(name)
print(f"训练集 trace 组合：{len(train_traces)} 个（将被排除）")

# ============ 计算 LRU 基准 IPC（排除训练集）============
lru_files = sorted(LRU_DIR.glob('*.csv'))
lru_ipc = {}
for f in lru_files:
    if f.name in train_traces:
        continue
    ipc = compute_ipc(f)
    if ipc is not None:
        lru_ipc[f.name] = ipc

print(f"LRU 基准：共 {len(lru_ipc)} 个有效 trace 组合（已排除训练集）")

# ============ 计算各策略归一化吞吐量 ============
throughputs = {name: {} for name in STRATEGIES}

for trace_name, base_ipc in lru_ipc.items():
    for strategy_name, strategy_dir in STRATEGIES.items():
        csv_path = resolve_strategy_path(strategy_dir, trace_name)
        if csv_path is None:
            continue
        ipc = compute_ipc(csv_path)
        if ipc is not None and base_ipc > 0:
            throughputs[strategy_name][trace_name] = ipc / base_ipc

# ============ 按 UCP 排序 ============
ucp_tp = throughputs['UCP']
common_traces = sorted(
    set.intersection(*(set(tp.keys()) for tp in throughputs.values())),
    key=lambda t: ucp_tp.get(t, 1.0)
)
print(f"三种策略共有的 trace 组合：{len(common_traces)} 个")

sorted_data = {}
for name in STRATEGIES:
    sorted_data[name] = [throughputs[name][t] for t in common_traces]

# ============ 绘图 ============
fig, ax = plt.subplots(figsize=(5.5, 3.0))

x = np.arange(len(common_traces))

COLORS = {
    'UCP':       '#e67e22',
    'Even Part': '#27ae60',
    'RL':        '#e74c3c',
}
STYLES = {
    'UCP':       {'linestyle': '-',  'linewidth': 1.0, 'zorder': 3},
    'Even Part': {'linestyle': ':',  'linewidth': 1.2, 'marker': '.', 'markersize': 2, 'zorder': 2},
    'RL':        {'linestyle': '--', 'linewidth': 1.0, 'zorder': 4},
}

for name in ['Even Part', 'UCP', 'RL']:
    ax.plot(x, sorted_data[name],
            color=COLORS[name], label=name, **STYLES[name])

ax.axhline(y=1.0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Workloads (sorted by UCP throughput)', fontweight='medium')
ax.set_ylabel('Norm. Throughput', fontweight='medium')

ax.set_xlim(0, len(common_traces) - 1)
ax.set_xticks([])

ax.grid(True, axis='y', linestyle='-', alpha=0.25, color='#555555')

ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc', ncol=3)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')

plt.tight_layout()
plt.savefig('plot/throughput_scurve_4core.png', dpi=300, facecolor='white')
plt.savefig('plot/throughput_scurve_4core.pdf', facecolor='white')
plt.close()

ucp_arr = np.array(sorted_data['UCP'])
even_arr = np.array(sorted_data['Even Part'])
rl_arr = np.array(sorted_data['RL'])

print(f"\n=== 统计摘要 ===")
for label, arr in [('UCP', ucp_arr), ('Even Part', even_arr), ('RL', rl_arr)]:
    print(f"{label:10s}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  "
          f">1.0: {(arr > 1.0).sum()}/{len(arr)}")

print(f"\n图像已保存为 plot/throughput_scurve_4core.png 和 .pdf")

# ============ 柱状图：统计摘要对比 ============
categories = ['Mean', 'Median', 'Min', 'Max']
bar_strategies = ['Even Part', 'UCP', 'RL']
BAR_COLORS = {
    'Even Part': '#1b7a3d',
    'UCP':       '#2ecc71',
    'RL':        '#a9dfbf',
}

stats_values = {}
for name, arr in [('Even Part', even_arr), ('UCP', ucp_arr), ('RL', rl_arr)]:
    stats_values[name] = [arr.mean(), np.median(arr), arr.min(), arr.max()]

fig2, ax2 = plt.subplots(figsize=(3.5, 3.0))

x_cat = np.arange(len(categories))
n_bars = len(bar_strategies)
bar_width = 0.22
offsets = np.arange(n_bars) - (n_bars - 1) / 2

for i, name in enumerate(bar_strategies):
    positions = x_cat + offsets[i] * bar_width
    ax2.bar(positions, stats_values[name], bar_width,
            color=BAR_COLORS[name], edgecolor='#333333', linewidth=0.6,
            label=name, zorder=3)

ax2.set_xticks(x_cat)
ax2.set_xticklabels(categories)
ax2.set_ylabel('Norm. Throughput', fontweight='medium')
ax2.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.6)

ax2.grid(True, axis='y', linestyle='-', alpha=0.25, color='#555555', zorder=0)
ax2.set_axisbelow(True)

ax2.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc')

for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')

plt.tight_layout()
plt.savefig('plot/throughput_stats_bar_4core.png', dpi=300, facecolor='white')
plt.savefig('plot/throughput_stats_bar_4core.pdf', facecolor='white')
plt.close()

print(f"柱状图已保存为 plot/throughput_stats_bar_4core.png 和 .pdf")
