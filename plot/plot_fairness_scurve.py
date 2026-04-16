#!/usr/bin/env python3
"""
绘制 2 核心配置下 Even / UCP / RL 三种策略的 Fairness S 曲线及统计柱状图。
Fairness = harmonic_mean(IPC_shared_cpu0 / IPC_alone_trace0,
                         IPC_shared_cpu1 / IPC_alone_trace1)
所有策略按 UCP 的 Fairness 从小到大排序。
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
ALONE_DIR = BASE_DIR / 'alone'
TRAIN_DIR = BASE_DIR / 'train'
STRATEGIES = {
    'Even Part': BASE_DIR / 'even_2core',
    'UCP':       BASE_DIR / 'ucp_2core',
    'RL':        BASE_DIR / 'rl_2core',
}


def compute_alone_ipc(csv_path: Path) -> float | None:
    """从 alone CSV 最后一行计算单独运行时的 IPC。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    last = df.iloc[-1]
    gc = last['global_cycle']
    if gc == 0:
        return None
    return last['instructions'] / gc


def compute_per_core_ipc(csv_path: Path) -> dict[int, float] | None:
    """从 2 核心 CSV 最后两行计算每个 CPU 的 IPC。返回 {cpu_id: ipc}。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if len(df) < 2:
        return None
    last_two = df.iloc[-2:]
    gc = last_two['global_cycle'].iloc[0]
    if gc == 0:
        return None
    return {int(row['cpu']): row['instructions'] / gc for _, row in last_two.iterrows()}


def parse_trace_names(filename: str) -> tuple[str, str] | None:
    """从文件名解析两个 trace 名称 (cpu0, cpu1)。"""
    stem = filename.replace('.csv', '')
    if stem.startswith('rl_'):
        stem = stem[3:]
    parts = stem.split('+')
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def resolve_strategy_path(strategy_dir: Path, canonical_csv: str) -> Path | None:
    """找到策略目录下对应的 CSV 文件（兼容 rl_ 前缀）。"""
    direct = strategy_dir / canonical_csv
    if direct.exists():
        return direct
    prefixed = strategy_dir / f"rl_{canonical_csv}"
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

# ============ 缓存 alone IPC ============
alone_ipc: dict[str, float] = {}
for f in ALONE_DIR.glob('*.csv'):
    ipc = compute_alone_ipc(f)
    if ipc is not None:
        alone_ipc[f.stem] = ipc

print(f"Alone 基准：{len(alone_ipc)} 个 trace")

# ============ 计算各策略的 Fairness ============
fairness: dict[str, dict[str, float]] = {name: {} for name in STRATEGIES}

ucp_dir = STRATEGIES['UCP']
for f in sorted(ucp_dir.glob('*.csv')):
    traces = parse_trace_names(f.name)
    if traces is None:
        continue
    trace0, trace1 = traces
    canonical = f"{trace0}+{trace1}.csv"
    if canonical in train_traces:
        continue
    if trace0 not in alone_ipc or trace1 not in alone_ipc:
        continue

    alone0, alone1 = alone_ipc[trace0], alone_ipc[trace1]

    for strat_name, strat_dir in STRATEGIES.items():
        csv_path = resolve_strategy_path(strat_dir, canonical)
        if csv_path is None:
            continue
        per_core = compute_per_core_ipc(csv_path)
        if per_core is None or 0 not in per_core or 1 not in per_core:
            continue

        r0 = per_core[0] / alone0
        r1 = per_core[1] / alone1
        if r0 > 0 and r1 > 0:
            fairness[strat_name][canonical] = 2.0 / (1.0 / r0 + 1.0 / r1)

# ============ 按 UCP Fairness 排序 ============
ucp_fair = fairness['UCP']
common_traces = sorted(
    set.intersection(*(set(v.keys()) for v in fairness.values())),
    key=lambda t: ucp_fair.get(t, 0.0),
)
print(f"三种策略共有的 trace 组合：{len(common_traces)} 个")

sorted_data = {name: [fairness[name][t] for t in common_traces] for name in STRATEGIES}

# ============ 绘图 ============
fig, ax = plt.subplots(figsize=(5.5, 3.0))
x = np.arange(len(common_traces))

COLORS = {
    'Even Part': '#27ae60',
    'UCP':       '#e67e22',
    'RL':        '#e74c3c',
}
STYLES = {
    'Even Part': {'linestyle': ':',  'linewidth': 1.2, 'marker': '.', 'markersize': 2, 'zorder': 2},
    'UCP':       {'linestyle': '-',  'linewidth': 1.0, 'zorder': 3},
    'RL':        {'linestyle': '--', 'linewidth': 1.0, 'zorder': 4},
}

for name in ['Even Part', 'UCP', 'RL']:
    ax.plot(x, sorted_data[name], color=COLORS[name], label=name, **STYLES[name])

ax.axhline(y=1.0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Workloads (sorted by UCP fairness)', fontweight='medium')
ax.set_ylabel('Fairness', fontweight='medium')
ax.set_xlim(0, len(common_traces) - 1)
ax.set_xticks([])

ax.grid(True, axis='y', linestyle='-', alpha=0.25, color='#555555')
ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc', ncol=3)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')

plt.tight_layout()
plt.savefig('plot/fairness_scurve_2core.png', dpi=300, facecolor='white')
plt.savefig('plot/fairness_scurve_2core.pdf', facecolor='white')
plt.close()

# ============ 统计摘要 ============
ucp_arr = np.array(sorted_data['UCP'])
even_arr = np.array(sorted_data['Even Part'])
rl_arr = np.array(sorted_data['RL'])

print(f"\n=== Fairness 统计摘要 ===")
for label, arr in [('Even Part', even_arr), ('UCP', ucp_arr), ('RL', rl_arr)]:
    print(f"{label:10s}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")

print(f"\nS 曲线已保存为 plot/fairness_scurve_2core.png 和 .pdf")

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
ax2.set_ylabel('Fairness', fontweight='medium')
ax2.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.6)

ax2.grid(True, axis='y', linestyle='-', alpha=0.25, color='#555555', zorder=0)
ax2.set_axisbelow(True)

ax2.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc')

for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')

plt.tight_layout()
plt.savefig('plot/fairness_stats_bar_2core.png', dpi=300, facecolor='white')
plt.savefig('plot/fairness_stats_bar_2core.pdf', facecolor='white')
plt.close()

print(f"柱状图已保存为 plot/fairness_stats_bar_2core.png 和 .pdf")
