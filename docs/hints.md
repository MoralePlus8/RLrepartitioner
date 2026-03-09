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

## 通过 **环境变量** 配置核心组，有两种方式

---

### 方式一：`RL_CORE_GROUPS` — 精确指定每个 CPU 的组归属

逗号分隔的组 ID 列表，按 CPU 编号顺序对应。

```bash
# 6 核心，分 3 组：CPU 0,1 → Group 0, CPU 2,3 → Group 1, CPU 4,5 → Group 2
RL_CORE_GROUPS=0,0,1,1,2,2 ./champsim --warmup 50000000 --simulation 200000000 trace0 trace1 trace2 trace3 trace4 trace5

# 4 核心，不均匀分组：CPU 0 独占一组，CPU 1,2,3 共享一组
RL_CORE_GROUPS=0,1,1,1 ./champsim --warmup 50000000 --simulation 200000000 trace0 trace1 trace2 trace3

# 4 核心，3 组（2+1+1）
RL_CORE_GROUPS=0,0,1,2 ./champsim ...
```

组数自动从最大组 ID 推断（`max_group_id + 1`）。如果提供的条目数少于 CPU 数，剩余 CPU 自动归入最后一组。

---

### 方式二：`RL_NUM_GROUPS` — 指定组数，自动均匀分配

```bash
# 6 核心均分 3 组：CPU 0,1 → G0, CPU 2,3 → G1, CPU 4,5 → G2
RL_NUM_GROUPS=3 ./champsim --warmup 50000000 --simulation 200000000 trace0 trace1 trace2 trace3 trace4 trace5

# 8 核心均分 4 组
RL_NUM_GROUPS=4 ./champsim ...
```

分配公式为 `cpu_id * num_groups / num_cpus`，尽量均匀。

---

### 不设置（默认行为）

```bash
# 默认 2 组，前半 CPU → Group 0，后半 → Group 1（与改动前行为一致）
./champsim --warmup 50000000 --simulation 200000000 trace0 trace1
```

---

### 优先级

`RL_CORE_GROUPS` > `RL_NUM_GROUPS` > 默认 2 组。两个变量同时设置时，以 `RL_CORE_GROUPS` 为准。

### 初始化输出验证

启动时日志会打印配置结果，可用于验证：

```
[RL_Partitioner] 替换策略初始化完成
  - 总 Way 数: 16
  - 核心组数: 3
  - Group 0: Way 数=6, CPUs=[0,1]
  - Group 1: Way 数=5, CPUs=[2,3]
  - Group 2: Way 数=5, CPUs=[4,5]
```

### 注意事项

- 组 ID 必须从 0 开始连续编号（例如不要写 `0,0,2,2` 跳过组 1，否则组 1 将存在但没有 CPU，会触发警告）
- 核心组数不能超过 LLC Way 数（每组至少需要 1 个 Way）
- 可以与其他 RL 环境变量组合使用，例如 `RL_CORE_GROUPS=0,0,1,1,2,2 RL_WEIGHTS_SAVE=weights.bin ./champsim ...`