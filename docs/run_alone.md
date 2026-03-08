# run_alone.py — 单核 LRU 基线模拟脚本

## 概述

`run_alone.py` 用于对 `traces/` 目录下的每个 trace 文件**逐一**运行单核 LRU 模拟器 (`bin/champsim_1core_lru`)，并将 LLC 统计结果以 CSV 格式输出到 `stats/alone/` 目录。

该脚本的典型用途是收集每个 benchmark 在**独占 LLC** 条件下的性能基线数据，作为后续多核共享 / RL 分区策略的对照参考。

## 前置条件

1. 已编译好单核 LRU 模拟器，可执行文件位于 `bin/champsim_1core_lru`。
2. `traces/` 目录下存放 `.champsimtrace.xz` 或 `.trace.xz` 格式的 trace 文件。
3. Python 3.6+（仅使用标准库，无需额外安装依赖）。

## 快速开始

```bash
# 预览所有任务（不实际执行）
python3 run_alone.py --dry-run

# 使用默认参数运行全部 trace
python3 run_alone.py

# 断点续跑：跳过已有结果
python3 run_alone.py --skip-existing
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bin` | `bin/champsim_1core_lru` | ChampSim 可执行文件路径 |
| `--traces-dir` | `traces/` | Trace 文件所在目录 |
| `--stats-dir` | `stats/alone/` | CSV 统计结果输出目录 |
| `--warmup` | `50,000,000` (50M) | 预热阶段指令数 |
| `--simulation` | `1,000,000,000` (1B) | 详细模拟阶段指令数 |
| `--workers` | `4` | 并行工作进程数 |
| `--skip-existing` | — | 跳过输出目录中已存在结果的 trace |
| `--dry-run` | — | 仅列出将要运行的任务，不实际执行 |

## 输出格式

每个 trace 生成一个独立的 CSV 文件，命名规则为：

```
stats/alone/<trace名>.csv
```

例如：

```
stats/alone/403.gcc-16B.csv
stats/alone/470.lbm-1274B.csv
stats/alone/605.mcf_s-472B.csv
```

## 运行示例

### 使用 8 个并行进程

```bash
python3 run_alone.py --workers 8
```

### 自定义模拟参数

```bash
python3 run_alone.py --warmup 10000000 --simulation 500000000
```

### 指定其他模拟器和输出目录

```bash
python3 run_alone.py --bin bin/champsim_1core_ship --stats-dir stats/alone_ship
```

## 运行流程

1. 扫描 `traces/` 目录，收集所有 trace 文件。
2. 为每个 trace 生成一条模拟命令：
   ```
   bin/champsim_1core_lru \
     --warmup-instructions 50000000 \
     --simulation-instructions 1000000000 \
     --csv-output stats/alone/<trace名>.csv \
     traces/<trace文件>
   ```
3. 通过 `ProcessPoolExecutor` 以指定并行度执行所有任务。
4. 实时打印进度、耗时及预估剩余时间。
5. 运行结束后汇总成功/失败数。

## 与 run_all_combinations.py 的区别

| | `run_alone.py` | `run_all_combinations.py` |
|---|---|---|
| 模拟器 | 单核 (`champsim_1core_lru`) | 双核 (`champsim_2core_rl`) |
| Trace 组合方式 | 每个 trace 独立运行 | 所有 trace 两两组合 C(n,2) |
| 任务数量 | n | n×(n-1)/2 |
| 用途 | 收集独占 LLC 基线数据 | 评估共享 LLC 下的 RL 分区策略 |
| 共享内存/权重 | 无 | 支持 mmap 共享 RL 权重 |
