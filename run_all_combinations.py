#!/usr/bin/env python3
"""
批量运行ChampSim模拟器，支持 2 核和 4 核配置
  - 2 核模式：遍历 traces 目录下所有 trace 文件的 C(n,2) 组合
  - 4 核模式：从 stats/trace_combinations_100.csv 读取预定义的 4-trace 组合
支持通过 mmap 共享内存实现多进程并行 RL 权重训练
"""

import os
import sys
import csv
import glob
import struct
import shutil
import subprocess
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import time

# ============================================================================
# 配置参数
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_CHAMPSIM_BIN_2CORE = PROJECT_ROOT / "bin" / "champsim_2core_rl"
DEFAULT_CHAMPSIM_BIN_4CORE = PROJECT_ROOT / "bin" / "champsim_4core_rl"
DEFAULT_TRACES_DIR = PROJECT_ROOT / "traces"
DEFAULT_STATS_DIR = PROJECT_ROOT / "stats"
DEFAULT_STATS_DIR_4CORE = PROJECT_ROOT / "stats" / "rl_4core"
DEFAULT_COMBINATIONS_CSV = PROJECT_ROOT / "stats" / "trace_combinations_100.csv"
DEFAULT_WARMUP = 50000000          # 50M
DEFAULT_SIMULATION = 500000000    # 500M
DEFAULT_WORKERS = 8
DEFAULT_WEIGHTS_FILE = str(PROJECT_ROOT / "rl_weights.bin")
DEFAULT_SHM_PATH = "/tmp/rl_shared_weights.shm"

# 与 C++ 侧 WeightsFileHeader / SharedWeightTable::Header 二进制兼容
HEADER_FORMAT = '<IIiiiIii'        # 8 × 4 = 32 bytes, little-endian packed
HEADER_MAGIC  = 0x524C5057         # "RLPW"
HEADER_VERSION = 1
RL_NUM_TILINGS = 8
RL_GRID_SIZE_A = 10
RL_GRID_SIZE_M = 10
RL_MEMORY_SIZE = 4096
RL_NUM_ACTIONS = 3
RL_TOTAL_WEIGHTS = RL_MEMORY_SIZE * RL_NUM_ACTIONS   # 12288

TRACE_SUFFIXES = [".champsimtrace.xz", ".trace.xz"]


# ============================================================================
# 共享内存管理
# ============================================================================

def create_shared_memory(shm_path: str, weights_file: str = None,
                         num_way: int = 16) -> bool:
    """
    创建共享内存文件，供多个 ChampSim 进程 mmap 映射。

    如果 weights_file 存在且格式正确，直接复制作为共享内存初始内容；
    否则创建全零初始化的新文件。
    """
    header_size = struct.calcsize(HEADER_FORMAT)
    expected_file_size = header_size + RL_TOTAL_WEIGHTS * 8

    if weights_file and os.path.exists(weights_file):
        file_size = os.path.getsize(weights_file)
        if file_size >= expected_file_size:
            # 校验 header
            with open(weights_file, 'rb') as f:
                raw = f.read(header_size)
            magic, version = struct.unpack_from('<II', raw, 0)
            if magic == HEADER_MAGIC and version == HEADER_VERSION:
                shutil.copy2(weights_file, shm_path)
                print(f"[SharedMemory] 已从 {weights_file} 初始化共享内存"
                      f" ({file_size} 字节)")
                return True
            else:
                print(f"[SharedMemory] 权重文件 {weights_file} 格式不正确，"
                      "将创建全零共享内存")
        else:
            print(f"[SharedMemory] 权重文件 {weights_file} 大小不足"
                  f" ({file_size} < {expected_file_size})，将创建全零共享内存")
    else:
        if weights_file:
            print(f"[SharedMemory] 权重文件 {weights_file} 不存在，"
                  "将创建全零共享内存")
        else:
            print("[SharedMemory] 未指定权重文件，将创建全零共享内存")

    # 创建全零初始化的共享内存文件
    with open(shm_path, 'wb') as f:
        header = struct.pack(
            HEADER_FORMAT,
            HEADER_MAGIC,
            HEADER_VERSION,
            RL_NUM_TILINGS,
            RL_GRID_SIZE_A,
            RL_GRID_SIZE_M,
            RL_MEMORY_SIZE,
            RL_NUM_ACTIONS,
            num_way
        )
        f.write(header)
        f.write(b'\x00' * (RL_TOTAL_WEIGHTS * 8))

    print(f"[SharedMemory] 已创建全零共享内存: {shm_path}"
          f" ({expected_file_size} 字节)")
    return True


def save_final_weights(shm_path: str, output_path: str) -> bool:
    """将共享内存中的权重保存为标准权重文件（用于下次训练的持久化）"""
    header_size = struct.calcsize(HEADER_FORMAT)
    expected_size = header_size + RL_TOTAL_WEIGHTS * 8

    if not os.path.exists(shm_path):
        print(f"[SharedMemory] 共享内存文件不存在: {shm_path}")
        return False

    file_size = os.path.getsize(shm_path)
    if file_size < expected_size:
        print(f"[SharedMemory] 共享内存文件大小不正确: {file_size}")
        return False

    shutil.copy2(shm_path, output_path)
    print(f"[SharedMemory] 最终权重已保存到: {output_path}")
    return True


# ============================================================================
# Trace 文件操作
# ============================================================================

def get_trace_files(traces_dir: Path) -> list:
    """获取traces目录下所有的trace文件"""
    patterns = [
        str(traces_dir / "*.champsimtrace.xz"),
        str(traces_dir / "*.trace.xz")
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def get_trace_name(trace_path: str) -> str:
    """从trace文件路径中提取简短名称（不含扩展名）"""
    basename = os.path.basename(trace_path)
    for suffix in TRACE_SUFFIXES:
        if basename.endswith(suffix):
            return basename[:-len(suffix)]
    return basename


def resolve_trace_path(trace_name: str, traces_dir: Path) -> str | None:
    """根据 trace 名称解析为 traces 目录下的完整路径"""
    for suffix in TRACE_SUFFIXES:
        path = traces_dir / f"{trace_name}{suffix}"
        if path.exists():
            return str(path)
    return None


def load_combinations_csv(csv_path: Path) -> list[tuple[int, list[str]]]:
    """
    从 CSV 加载 4-trace 组合，返回 (combo_id, [trace1, trace2, trace3, trace4]) 列表。
    使用标准库 csv 模块，无需 pandas。
    """
    combos = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        required = ["combo_id", "trace_1", "trace_2", "trace_3", "trace_4"]
        if reader.fieldnames is None:
            print(f"错误: CSV 文件为空: {csv_path}", file=sys.stderr)
            sys.exit(1)
        for col in required:
            if col not in reader.fieldnames:
                print(f"错误: CSV 缺少列 '{col}'", file=sys.stderr)
                sys.exit(1)
        for row in reader:
            combo_id = int(row["combo_id"])
            traces = [row["trace_1"], row["trace_2"],
                      row["trace_3"], row["trace_4"]]
            combos.append((combo_id, traces))
    return combos


# ============================================================================
# 模拟任务
# ============================================================================

def run_simulation(args_tuple: tuple) -> dict:
    """
    运行单个模拟任务（支持 2 核或 4 核）

    Args:
        args_tuple: (traces_list, output_csv, champsim_bin,
                     warmup, simulation, extra_env) 元组
    """
    (traces, output_csv, champsim_bin,
     warmup, simulation, extra_env) = args_tuple

    trace_names = [get_trace_name(t) for t in traces]

    result = {
        "traces": " + ".join(trace_names),
        "output": output_csv,
        "success": False,
        "error": None,
        "duration": 0
    }

    start_time = time.time()

    cmd = [
        str(champsim_bin),
        "--warmup-instructions", str(warmup),
        "--simulation-instructions", str(simulation),
        "--csv-output", output_csv,
        *traces
    ]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    try:
        process = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=86400 # 24 hours
        )

        if os.path.exists(output_csv):
            result["success"] = True
        else:
            result["error"] = (f"CSV file not generated. "
                               f"Return code: {process.returncode}")
            if process.stderr:
                result["error"] += f"\nStderr: {process.stderr[:500]}"

    except subprocess.TimeoutExpired:
        result["error"] = "Simulation timed out (>24 hours)"
    except Exception as e:
        result["error"] = str(e)

    result["duration"] = time.time() - start_time
    return result


def generate_task_list_2core(traces: list, stats_dir: Path, champsim_bin: Path,
                             warmup: int, simulation: int,
                             extra_env: dict = None,
                             csv_prefix: str = "rl_") -> list:
    """生成 2 核 C(n,2) 任务列表"""
    tasks = []
    for trace1, trace2 in itertools.combinations(traces, 2):
        name1 = get_trace_name(trace1)
        name2 = get_trace_name(trace2)
        output_csv = str(stats_dir / f"{csv_prefix}{name1}+{name2}.csv")
        tasks.append(([trace1, trace2], output_csv, str(champsim_bin),
                       warmup, simulation, extra_env))
    return tasks


def generate_task_list_4core(combinations_csv: Path, traces_dir: Path,
                             stats_dir: Path, champsim_bin: Path,
                             warmup: int, simulation: int,
                             extra_env: dict = None,
                             csv_prefix: str = "rl_") -> list:
    """从 CSV 读取 4-trace 组合，生成 4 核任务列表"""
    combos = load_combinations_csv(combinations_csv)
    tasks = []
    for combo_id, trace_names in combos:
        paths = []
        for name in trace_names:
            p = resolve_trace_path(name, traces_dir)
            if p is None:
                print(f"警告: 跳过 combo {combo_id}，未找到 trace: {name}")
                break
            paths.append(p)
        if len(paths) == 4:
            output_name = f"{csv_prefix}{'+'.join(trace_names)}.csv"
            output_csv = str(stats_dir / output_name)
            tasks.append((paths, output_csv, str(champsim_bin),
                           warmup, simulation, extra_env))
    return tasks


def filter_existing_tasks(tasks: list) -> list:
    """过滤掉已经存在结果的任务"""
    filtered = []
    for task in tasks:
        output_csv = task[1]  # traces_list, output_csv, ...
        if not os.path.exists(output_csv):
            filtered.append(task)
    return filtered


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="批量运行ChampSim模拟器 (支持 2 核 / 4 核，共享内存 RL 权重训练)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 2 核模式 (默认)
  python run_all_combinations.py --dry-run                    # 预览任务
  python run_all_combinations.py --limit 10                   # 只运行10个任务
  python run_all_combinations.py --skip-existing              # 跳过已有结果

  # 4 核模式
  python run_all_combinations.py --cores 4 --dry-run
  python run_all_combinations.py --cores 4 --skip-existing
  python run_all_combinations.py --cores 4 --combinations-csv stats/trace_combinations_100.csv

  # 通用选项
  python run_all_combinations.py --workers 4                  # 使用4个并行进程
  python run_all_combinations.py --no-shared-weights          # 禁用共享内存
        """
    )
    parser.add_argument(
        "--cores",
        type=int,
        choices=[2, 4],
        default=2,
        help="核心数配置：2 = 两两组合, 4 = 从 CSV 读取四核组合 (默认: 2)"
    )
    parser.add_argument(
        "--bin",
        type=str,
        default=None,
        help=("ChampSim 可执行文件路径 "
              f"(默认: 2核={DEFAULT_CHAMPSIM_BIN_2CORE}, "
              f"4核={DEFAULT_CHAMPSIM_BIN_4CORE})")
    )
    parser.add_argument(
        "--combinations-csv",
        type=str,
        default=str(DEFAULT_COMBINATIONS_CSV),
        help=f"4 核模式的组合列表 CSV 路径 (默认: {DEFAULT_COMBINATIONS_CSV})"
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default=str(DEFAULT_TRACES_DIR),
        help=f"Traces文件目录路径 (默认: {DEFAULT_TRACES_DIR})"
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=None,
        help=("统计结果输出目录 "
              f"(默认: 2核={DEFAULT_STATS_DIR}, "
              f"4核={DEFAULT_STATS_DIR_4CORE})")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行工作进程数 (默认: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"预热指令数 (默认: {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "--simulation",
        type=int,
        default=DEFAULT_SIMULATION,
        help=f"模拟指令数 (默认: {DEFAULT_SIMULATION})"
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default=DEFAULT_WEIGHTS_FILE,
        help=f"权重文件路径，用于初始加载和最终保存 (默认: {DEFAULT_WEIGHTS_FILE})"
    )
    parser.add_argument(
        "--shm-path",
        type=str,
        default=DEFAULT_SHM_PATH,
        help=f"共享内存文件路径 (默认: {DEFAULT_SHM_PATH})"
    )
    parser.add_argument(
        "--num-way",
        type=int,
        default=16,
        help="LLC Way 数量，须与 ChampSim 配置一致 (默认: 16)"
    )
    parser.add_argument(
        "--no-shared-weights",
        action="store_true",
        help="禁用共享内存，各进程使用文件 load/save 独立训练"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="纯评估模式：加载权重但不保存、不更新共享内存（只读评估）"
    )
    parser.add_argument(
        "--csv-prefix",
        type=str,
        default="rl_",
        help="CSV 输出文件名前缀 (默认: rl_)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在的结果文件"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要运行的任务，不实际执行"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制运行的任务数量（用于测试）"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 根据核心数解析默认值
    # ------------------------------------------------------------------
    if args.bin is None:
        champsim_bin = (DEFAULT_CHAMPSIM_BIN_4CORE if args.cores == 4
                        else DEFAULT_CHAMPSIM_BIN_2CORE)
    else:
        champsim_bin = Path(args.bin)

    if args.stats_dir is None:
        stats_dir = (DEFAULT_STATS_DIR_4CORE if args.cores == 4
                     else DEFAULT_STATS_DIR)
    else:
        stats_dir = Path(args.stats_dir)

    traces_dir = Path(args.traces_dir)
    eval_mode = args.eval
    use_shared = not args.no_shared_weights and not eval_mode

    # ------------------------------------------------------------------
    # 检查前置条件
    # ------------------------------------------------------------------
    if not champsim_bin.exists():
        print(f"错误: ChampSim 可执行文件不存在: {champsim_bin}")
        print("请先编译: ./config.sh champsim_config.json && make")
        sys.exit(1)

    if not traces_dir.exists():
        print(f"错误: Traces 目录不存在: {traces_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 构造子进程环境变量
    # ------------------------------------------------------------------
    extra_env = {}

    if eval_mode:
        extra_env["RL_WEIGHTS_LOAD"] = args.weights_file
        extra_env["RL_ONLINE_MODE"] = "1"
    else:
        extra_env["RL_WEIGHTS_SAVE"] = args.weights_file
        extra_env["RL_WEIGHTS_LOAD"] = args.weights_file
        if use_shared:
            extra_env["RL_SHARED_WEIGHTS"] = args.shm_path

    # ------------------------------------------------------------------
    # 生成任务列表
    # ------------------------------------------------------------------
    if args.cores == 4:
        combinations_csv = Path(args.combinations_csv)
        if not combinations_csv.exists():
            print(f"错误: 组合列表 CSV 不存在: {combinations_csv}")
            sys.exit(1)

        tasks = generate_task_list_4core(
            combinations_csv, traces_dir, stats_dir, champsim_bin,
            args.warmup, args.simulation,
            extra_env=extra_env, csv_prefix=args.csv_prefix)
        print(f"从 {combinations_csv} 加载 {len(tasks)} 个有效的 4 核组合")
    else:
        traces = get_trace_files(traces_dir)
        if not traces:
            print(f"错误: 在 {traces_dir} 目录下未找到 trace 文件")
            sys.exit(1)
        print(f"找到 {len(traces)} 个 trace 文件")

        tasks = generate_task_list_2core(
            traces, stats_dir, champsim_bin,
            args.warmup, args.simulation,
            extra_env=extra_env, csv_prefix=args.csv_prefix)
        n = len(traces)
        print(f"共有 {len(tasks)} 个组合 (C({n},2) = {n}*{n-1}//2)")

    total_combinations = len(tasks)

    if args.skip_existing:
        tasks = filter_existing_tasks(tasks)
        skipped = total_combinations - len(tasks)
        if skipped > 0:
            print(f"跳过 {skipped} 个已存在的结果文件")

    if args.limit:
        tasks = tasks[:args.limit]
        print(f"限制运行 {len(tasks)} 个任务")

    if not tasks:
        print("没有任务需要运行")
        return

    os.makedirs(stats_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    if args.dry_run:
        print(f"\n将要运行的任务 (共 {len(tasks)} 个):")
        print("-" * 70)
        for i, task in enumerate(tasks[:20]):
            trace_names = [get_trace_name(t) for t in task[0]]
            print(f"  {i+1:4d}. {' + '.join(trace_names)}")
        if len(tasks) > 20:
            print(f"  ... 还有 {len(tasks) - 20} 个任务")
        print("-" * 70)
        print(f"核心数:       {args.cores}")
        print(f"输出目录:     {stats_dir}")
        print(f"可执行文件:   {champsim_bin}")
        print(f"并行进程数:   {args.workers}")
        mode_str = "纯评估（只读，不保存权重）" if eval_mode else (
            "训练 + 共享内存 (" + args.shm_path + ")" if use_shared
            else "训练（独立，无共享内存）")
        print(f"运行模式:     {mode_str}")
        print(f"权重文件:     {args.weights_file}")
        print(f"Warmup:       {args.warmup:,} 指令")
        print(f"Simulation:   {args.simulation:,} 指令")
        if args.cores == 4:
            print(f"组合来源:     {args.combinations_csv}")
        return

    # ------------------------------------------------------------------
    # 初始化共享内存
    # ------------------------------------------------------------------
    if use_shared:
        print("\n初始化共享内存权重表...")
        create_shared_memory(args.shm_path, args.weights_file, args.num_way)

    # ------------------------------------------------------------------
    # 运行模拟
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"使用 {args.workers} 个进程并行运行 {len(tasks)} 个 {args.cores} 核模拟任务")
    if eval_mode:
        print(f"模式: 纯评估（只读权重，不保存）")
    elif use_shared:
        print(f"模式: 训练 + 共享内存 ({args.shm_path})")
    else:
        print(f"模式: 训练（独立，无共享内存）")
    print(f"权重文件: {args.weights_file}")
    print(f"{'='*70}")

    start_time = time.time()
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(run_simulation, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            result = future.result()
            completed += 1

            if result["success"]:
                status = "✓"
            else:
                status = "✗"
                failed += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            remaining = (len(tasks) - completed) * avg_time / max(args.workers, 1)

            print(f"[{completed:4d}/{len(tasks)}] {status} "
                  f"{result['traces']} "
                  f"({result['duration']:.1f}s) "
                  f"- 预计剩余: {remaining/60:.1f}分钟")

            if not result["success"]:
                print(f"         错误: {result['error']}")

    # ------------------------------------------------------------------
    # 保存最终权重 & 清理
    # ------------------------------------------------------------------
    if use_shared:
        print("\n保存最终权重...")
        save_final_weights(args.shm_path, args.weights_file)
        print(f"(共享内存文件 {args.shm_path} 已保留，可手动删除)")

    # 输出统计
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"完成！总用时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"成功: {completed - failed}, 失败: {failed}")
    print(f"结果保存在: {stats_dir}")
    if not eval_mode:
        print(f"权重保存在: {args.weights_file}")
    else:
        print(f"纯评估模式，权重未修改")


if __name__ == "__main__":
    main()
