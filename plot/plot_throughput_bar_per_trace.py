#!/usr/bin/env python3
"""
绘制柱状图：每个 trace 在所有 2 核心组合下 Even / UCP / RL 三种策略
相对于 LRU 基准的平均归一化吞吐量，按 RL 排序。
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

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

BASE_DIR = Path('stats')
LRU_DIR = BASE_DIR / 'lru_2core'

STRATEGIES = {
    'Even Part': BASE_DIR / 'even_2core',
    'UCP':       BASE_DIR / 'ucp_2core',
    'RL':        BASE_DIR / 'rl_2core',
}


def extract_trace_name(raw: str) -> str:
    """Remove trailing '-xxxB' suffix to get the clean trace name."""
    return re.sub(r'-\d+B$', '', raw)


def get_per_cpu_ipc(csv_path: Path) -> dict[int, float] | None:
    """Read a CSV and return {cpu_id: ipc} from the last two rows."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if len(df) < 2:
        return None
    last_two = df.iloc[-2:]
    result = {}
    for _, row in last_two.iterrows():
        cpu = int(row['cpu'])
        instructions = row['instructions']
        global_cycle = row['global_cycle']
        if global_cycle == 0:
            continue
        result[cpu] = instructions / global_cycle
    return result if len(result) == 2 else None


def resolve_strategy_file(strategy_dir: Path, stem: str) -> Path | None:
    """Find the CSV: try direct name, then with rl_ prefix."""
    direct = strategy_dir / f"{stem}.csv"
    if direct.exists():
        return direct
    prefixed = strategy_dir / f"rl_{stem}.csv"
    if prefixed.exists():
        return prefixed
    return None


# ============ 计算各策略各 trace 的归一化吞吐量 ============
throughputs_by_trace = {name: defaultdict(list) for name in STRATEGIES}

lru_files = sorted(LRU_DIR.glob('*.csv'))
for lru_file in lru_files:
    stem = lru_file.stem
    parts = stem.split('+')
    if len(parts) != 2:
        continue
    trace0_name = extract_trace_name(parts[0])
    trace1_name = extract_trace_name(parts[1])

    lru_ipc = get_per_cpu_ipc(lru_file)
    if lru_ipc is None:
        continue

    for strategy_name, strategy_dir in STRATEGIES.items():
        csv_path = resolve_strategy_file(strategy_dir, stem)
        if csv_path is None:
            continue
        s_ipc = get_per_cpu_ipc(csv_path)
        if s_ipc is None:
            continue
        if lru_ipc[0] > 0:
            throughputs_by_trace[strategy_name][trace0_name].append(
                s_ipc[0] / lru_ipc[0])
        if lru_ipc[1] > 0:
            throughputs_by_trace[strategy_name][trace1_name].append(
                s_ipc[1] / lru_ipc[1])

mean_tp = {
    name: {t: np.mean(v) for t, v in tdict.items()}
    for name, tdict in throughputs_by_trace.items()
}

sorted_traces = sorted(mean_tp['RL'], key=lambda t: mean_tp['RL'][t])

print(f"共 {len(sorted_traces)} 个 trace")
for t in sorted_traces:
    vals = '  '.join(f"{name}={mean_tp[name].get(t, float('nan')):.4f}"
                     for name in STRATEGIES)
    print(f"  {t:25s}  {vals}")

# ============ 绘图 ============
fig, ax = plt.subplots(figsize=(5.5, 3.0))

n_traces = len(sorted_traces)
n_bars = len(STRATEGIES)
bar_width = 0.25
group_width = bar_width * n_bars
x = np.arange(n_traces)

BAR_COLORS = {
    'Even Part': '#1b7a3d',
    'UCP':       '#2ecc71',
    'RL':        '#a9dfbf',
}

for i, name in enumerate(STRATEGIES):
    values = [mean_tp[name].get(t, 0.0) for t in sorted_traces]
    offset = (i - (n_bars - 1) / 2) * bar_width
    ax.bar(x + offset, values, bar_width,
           color=BAR_COLORS[name], edgecolor='#333333', linewidth=0.4,
           zorder=3, label=name)

ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.6)

ax.set_xlabel('Traces (sorted by RL Norm. Throughput)', fontweight='medium')
ax.set_ylabel('Norm. Throughput', fontweight='medium')

ax.set_xticks(x)
ax.set_xticklabels(sorted_traces, rotation=60, ha='right', fontsize=5.5)
ax.set_xlim(-0.5, n_traces - 0.5)

ax.grid(True, axis='y', linestyle='-', alpha=0.25, color='#555555', zorder=0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')

ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc', ncol=3)

plt.tight_layout()
plt.savefig('plot/throughput_bar_per_trace_2core.png', dpi=300, facecolor='white')
plt.savefig('plot/throughput_bar_per_trace_2core.pdf', facecolor='white')
plt.close()

print(f"\n=== 统计摘要 ===")
for name in STRATEGIES:
    arr = np.array([mean_tp[name].get(t, 0.0) for t in sorted_traces])
    print(f"  {name:10s}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  "
          f">1.0: {(arr > 1.0).sum()}/{len(arr)}")
print(f"\n图像已保存为 plot/throughput_bar_per_trace_2core.png 和 .pdf")
