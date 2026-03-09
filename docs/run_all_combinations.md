# run_all_combinations.py 使用指南

批量运行 ChampSim 模拟器，自动遍历 traces 目录中所有 trace 文件的两两组合（C(n,2) 个），在多进程并行执行的同时，通过 mmap 共享内存实现 RL 权重的实时跨进程共享训练。

## 前置条件

1. 已编译 ChampSim（含 RL 分区策略）：

```bash
./config.sh champsim_config.json
make -j$(nproc)
```

2. traces 目录中至少放置 2 个 trace 文件（支持 `*.champsimtrace.xz` 和 `*.trace.xz` 格式）。

## 快速开始

```bash
# 使用默认参数运行全部组合（8 进程并行，共享内存训练）
python run_all_combinations.py

# 先预览任务列表，不实际执行
python run_all_combinations.py --dry-run
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bin` | `bin/champsim` | ChampSim 可执行文件路径 |
| `--traces-dir` | `traces/` | trace 文件所在目录 |
| `--stats-dir` | `stats/` | CSV 统计结果输出目录 |
| `--workers` | `8` | 并行工作进程数 |
| `--warmup` | `50000000` (50M) | 预热阶段指令数 |
| `--simulation` | `1000000000` (1B) | 模拟阶段指令数 |
| `--weights-file` | `rl_weights.bin` | 权重文件路径（初始加载 + 最终保存） |
| `--shm-path` | `/tmp/rl_shared_weights.shm` | 共享内存文件路径 |
| `--num-way` | `16` | LLC Way 数，须与 ChampSim 配置一致 |
| `--csv-prefix` | `rl_` | CSV 输出文件名前缀 |
| `--no-shared-weights` | — | 禁用共享内存，各进程独立训练 |
| `--skip-existing` | — | 跳过已有 CSV 结果的任务 |
| `--dry-run` | — | 仅打印任务列表，不执行 |
| `--limit N` | — | 只运行前 N 个任务（用于测试） |
| `--eval` | — | 评估模式，不保存权重文件且禁用共享内存 |

## 使用示例

### 1. 完整训练（默认共享内存模式）

```bash
python run_all_combinations.py
```

自动完成：
- 扫描 `traces/` 下所有 trace → 生成 C(n,2) 个组合
- 创建共享内存文件（若 `rl_weights.bin` 存在则加载，否则全零初始化）
- 8 个进程并行模拟，实时共享权重
- 每个进程完成时保存权重快照到 `rl_weights.bin`
- 全部完成后将共享内存最终状态保存到 `rl_weights.bin`

### 2. 从已有权重继续训练

```bash
python run_all_combinations.py --weights-file ./pretrained_weights.bin
```

脚本会将 `pretrained_weights.bin` 的内容复制到共享内存作为初始权重，训练完成后再保存回该文件。

### 3. 小规模测试

```bash
# 只跑 5 个组合，2 个进程，缩短模拟时间
python run_all_combinations.py --limit 5 --workers 2 --simulation 100000000
```

### 4. 跳过已完成的任务（断点续跑）

```bash
python run_all_combinations.py --skip-existing
```

仅运行 `stats/` 目录中尚未生成 CSV 文件的组合。

### 5. 禁用共享内存（各进程独立训练）

```bash
python run_all_combinations.py --no-shared-weights
```

此模式下每个进程：
- 启动时从 `rl_weights.bin` 加载权重（如果文件存在）
- 结束时将训练结果保存回 `rl_weights.bin`
- 进程之间不共享权重，最终文件内容取决于最后完成的进程

### 6. 自定义输出

```bash
python run_all_combinations.py \
  --stats-dir results/round2 \
  --csv-prefix "round2_" \
  --weights-file weights/round2.bin
```

## 工作原理

### 共享内存训练流程

```
脚本启动
  │
  ├─ 创建共享内存文件 (/tmp/rl_shared_weights.shm)
  │   └─ 若 rl_weights.bin 存在 → 复制内容; 否则 → 全零初始化
  │
  ├─ 启动 8 个 ChampSim 子进程
  │   ├─ 进程 1: trace_A + trace_B ──┐
  │   ├─ 进程 2: trace_A + trace_C   │  全部 mmap 同一个共享内存文件
  │   ├─ ...                          │  通过 flock 读写锁保护并发访问
  │   └─ 进程 8: trace_D + trace_E ──┘
  │
  │   每个进程内部，每 500K cycles:
  │     1. read_lock  → 从共享内存拉取最新权重
  │     2. 本地计算: train + experience replay（无锁）
  │     3. write_lock → 将权重增量推送回共享内存
  │
  │   进程完成模拟 → 保存权重快照到 rl_weights.bin
  │
  ├─ 所有进程完成后
  │   └─ 将共享内存最终状态保存到 rl_weights.bin
  │
  └─ 输出统计信息
```

### 环境变量

脚本自动为每个子进程设置以下环境变量：

| 环境变量 | 说明 |
|----------|------|
| `RL_SHARED_WEIGHTS` | 共享内存文件路径（共享内存模式下设置） |
| `RL_WEIGHTS_SAVE` | 权重保存路径（模拟结束时写入） |
| `RL_WEIGHTS_LOAD` | 权重加载路径（共享内存连接失败时的回退） |

### 输出文件

- **CSV 统计文件**: `stats/rl_<trace1>+<trace2>.csv` — 每个 trace 组合的 LLC 统计数据
- **权重文件**: `rl_weights.bin` — 训练得到的 Q-learning 权重（二进制格式）
- **共享内存文件**: `/tmp/rl_shared_weights.shm` — 训练完成后保留，可手动删除

## 常见问题

**Q: 首次运行时没有 `rl_weights.bin` 怎么办？**

脚本会自动创建全零初始化的共享内存，从头开始训练。控制台会提示：

```
[SharedMemory] 权重文件 rl_weights.bin 不存在，将创建全零共享内存
```

**Q: 运行中断后如何恢复？**

使用 `--skip-existing` 跳过已完成的组合。由于每个进程结束时都会保存权重，`rl_weights.bin` 中保留了中断前的训练进度：

```bash
python run_all_combinations.py --skip-existing
```

**Q: 如何调整并行度？**

根据机器的 CPU 核心数和内存调整 `--workers`。每个 ChampSim 进程约占 1-2 GB 内存：

```bash
# 4 核机器
python run_all_combinations.py --workers 4
```

**Q: 共享内存文件没有被清理怎么办？**

训练完成后脚本会保留 `/tmp/rl_shared_weights.shm` 以备调试，可以手动删除：

```bash
rm /tmp/rl_shared_weights.shm
```
