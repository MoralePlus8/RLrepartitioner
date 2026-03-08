### 离线训练参数

**三个环境变量控制全部行为：**

| 环境变量 | 作用 | 示例值 |
|---------|------|--------|
| `RL_WEIGHTS_SAVE` | 模拟结束后保存权重的路径 | `./rl_weights.bin` |
| `RL_WEIGHTS_LOAD` | 模拟开始时加载权重的路径 | `./rl_weights.bin` |
| `RL_ONLINE_MODE` | 设为 `1` 时启用在线微调（LR=0.02, ε=0.05）| `1` |

**场景一：离线预训练（在多组 trace 上串行累积训练）**

```bash
# 第 1 组 trace: 从零开始训练，结束后保存
RL_WEIGHTS_SAVE=./rl_weights.bin \
  ./champsim --warmup 50000000 --simulation 200000000 \
  traces/403.gcc-16B.champsimtrace.xz traces/470.lbm-1274B.champsimtrace.xz

# 第 2 组 trace: 加载上轮权重，继续训练并保存
RL_WEIGHTS_LOAD=./rl_weights.bin RL_WEIGHTS_SAVE=./rl_weights.bin \
  ./champsim --warmup 50000000 --simulation 200000000 \
  traces/429.mcf-22B.champsimtrace.xz traces/456.hmmer-327B.champsimtrace.xz

# ... 更多 trace 组合重复上面的命令
```

**场景二：在线微调（用预训练权重跑新 trace）**

```bash
RL_WEIGHTS_LOAD=./rl_weights.bin RL_ONLINE_MODE=1 RL_WEIGHTS_SAVE=./rl_weights_finetuned.bin \
  ./champsim --warmup 50000000 --simulation 200000000 \
  traces/605.mcf_s-472B.champsimtrace.xz traces/619.lbm_s-2676B.champsimtrace.xz
```

**场景三：纯评估（加载权重但不保存）**

```bash
RL_WEIGHTS_LOAD=./rl_weights.bin RL_ONLINE_MODE=1 \
  ./champsim --warmup 50000000 --simulation 200000000 \
  traces/603.bwaves_s-1740B.champsimtrace.xz traces/607.cactuBSSN_s-2421B.champsimtrace.xz
```

不设置任何环境变量时，行为与之前完全一致（从零开始训练，不保存）。

### 统计数据输出路径

在运行模拟时，加上 `--csv-output` 参数即可指定输出路径，例如：

```bash
./bin/champsim --warmup-instructions 1000000 --simulation-instructions 10000000 \
  --csv-output stats/my_output.csv \
  traces/trace1.champsimtrace.xz traces/trace2.champsimtrace.xz
```

如果不指定 `--csv-output`，则默认输出到 `llc_stats.csv`（由 `g_llc_stats_csv_path` 的默认值决定）。