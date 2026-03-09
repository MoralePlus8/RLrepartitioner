#!/usr/bin/env python3
"""
读取 traces 目录下的所有 trace 文件，两两组合，
每组使用 bin/champsim_2core_even 模拟运行，
将统计数据文件输出到 stats/even_2core 目录。
"""

import os
import sys
import glob
import time
import subprocess
import argparse
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_BIN = PROJECT_ROOT / "bin" / "champsim_2core_even"
DEFAULT_TRACES_DIR = PROJECT_ROOT / "traces"
DEFAULT_STATS_DIR = PROJECT_ROOT / "stats" / "even_2core"
DEFAULT_WARMUP = 50_000_000       # 50M
DEFAULT_SIMULATION = 500_000_000  # 500M
DEFAULT_WORKERS = 8


def get_trace_files(traces_dir: Path) -> list:
    """获取 traces 目录下所有 trace 文件"""
    patterns = [
        str(traces_dir / "*.champsimtrace.xz"),
        str(traces_dir / "*.trace.xz"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))


def get_trace_name(trace_path: str) -> str:
    """从 trace 路径提取简短名称（不含扩展名）"""
    basename = os.path.basename(trace_path)
    for suffix in [".champsimtrace.xz", ".trace.xz"]:
        if basename.endswith(suffix):
            return basename[: -len(suffix)]
    return basename


def run_single(args_tuple: tuple) -> dict:
    """运行单组两核模拟任务"""
    trace1, trace2, output_csv, champsim_bin, warmup, simulation = args_tuple
    name1 = get_trace_name(trace1)
    name2 = get_trace_name(trace2)

    result = {
        "trace1": name1,
        "trace2": name2,
        "output": output_csv,
        "success": False,
        "error": None,
        "duration": 0,
    }

    start = time.time()

    cmd = [
        str(champsim_bin),
        "--warmup-instructions", str(warmup),
        "--simulation-instructions", str(simulation),
        "--csv-output", output_csv,
        trace1,
        trace2,
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=43200,
        )
        if os.path.exists(output_csv):
            result["success"] = True
        else:
            result["error"] = (
                f"CSV not generated. Return code: {proc.returncode}"
            )
            if proc.stderr:
                result["error"] += f"\nStderr: {proc.stderr[:500]}"
    except subprocess.TimeoutExpired:
        result["error"] = "Timed out (>12h)"
    except Exception as e:
        result["error"] = str(e)

    result["duration"] = time.time() - start
    return result


def main():
    parser = argparse.ArgumentParser(
        description="两核 Even 分区模拟：traces 两两组合运行 ChampSim",
    )
    parser.add_argument(
        "--bin", type=str, default=str(DEFAULT_BIN),
        help=f"可执行文件路径 (默认: {DEFAULT_BIN})",
    )
    parser.add_argument(
        "--traces-dir", type=str, default=str(DEFAULT_TRACES_DIR),
        help=f"Traces 目录 (默认: {DEFAULT_TRACES_DIR})",
    )
    parser.add_argument(
        "--stats-dir", type=str, default=str(DEFAULT_STATS_DIR),
        help=f"输出目录 (默认: {DEFAULT_STATS_DIR})",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"预热指令数 (默认: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--simulation", type=int, default=DEFAULT_SIMULATION,
        help=f"模拟指令数 (默认: {DEFAULT_SIMULATION})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"并行进程数 (默认: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="跳过已存在结果的组合",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只显示将运行的任务，不实际执行",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="限制运行的任务数量（用于测试）",
    )

    args = parser.parse_args()

    champsim_bin = Path(args.bin)
    traces_dir = Path(args.traces_dir)
    stats_dir = Path(args.stats_dir)

    if not champsim_bin.exists():
        print(f"错误: 可执行文件不存在: {champsim_bin}")
        sys.exit(1)

    if not traces_dir.exists():
        print(f"错误: Traces 目录不存在: {traces_dir}")
        sys.exit(1)

    traces = get_trace_files(traces_dir)
    if not traces:
        print(f"错误: 未找到 trace 文件: {traces_dir}")
        sys.exit(1)

    print(f"找到 {len(traces)} 个 trace 文件")

    # 生成 C(n,2) 两两组合
    tasks = []
    for trace1, trace2 in itertools.combinations(traces, 2):
        name1 = get_trace_name(trace1)
        name2 = get_trace_name(trace2)
        output_csv = str(stats_dir / f"{name1}+{name2}.csv")
        tasks.append((trace1, trace2, output_csv, str(champsim_bin),
                      args.warmup, args.simulation))

    total_combinations = len(tasks)
    print(f"共有 {total_combinations} 个组合 (C({len(traces)},2))")

    if args.skip_existing:
        before = len(tasks)
        tasks = [t for t in tasks if not os.path.exists(t[2])]
        skipped = before - len(tasks)
        if skipped:
            print(f"跳过 {skipped} 个已存在的结果")

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"限制运行 {len(tasks)} 个任务")

    if not tasks:
        print("没有任务需要运行")
        return

    os.makedirs(stats_dir, exist_ok=True)

    if args.dry_run:
        print(f"\n将运行 {len(tasks)} 个任务:")
        print("-" * 70)
        for i, t in enumerate(tasks[:20]):
            print(f"  {i+1:4d}. {get_trace_name(t[0])} + {get_trace_name(t[1])}")
        if len(tasks) > 20:
            print(f"  ... 还有 {len(tasks) - 20} 个任务")
        print("-" * 70)
        print(f"可执行文件: {champsim_bin}")
        print(f"输出目录:   {stats_dir}")
        print(f"并行进程数: {args.workers}")
        print(f"Warmup:     {args.warmup:,} 指令")
        print(f"Simulation: {args.simulation:,} 指令")
        return

    print(f"\n{'='*70}")
    print(f"使用 {args.workers} 个进程并行运行 {len(tasks)} 个模拟任务")
    print(f"输出目录: {stats_dir}")
    print(f"{'='*70}")

    start_time = time.time()
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(run_single, task): task for task in tasks
        }
        for future in as_completed(future_map):
            result = future.result()
            completed += 1

            if result["success"]:
                status = "OK"
            else:
                status = "FAIL"
                failed += 1

            elapsed = time.time() - start_time
            avg = elapsed / completed
            remaining = (len(tasks) - completed) * avg / max(args.workers, 1)

            print(
                f"[{completed:4d}/{len(tasks)}] {status:4s} "
                f"{result['trace1']} + {result['trace2']}  ({result['duration']:.1f}s)  "
                f"预计剩余: {remaining/60:.1f}min"
            )
            if not result["success"]:
                print(f"         错误: {result['error']}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"完成! 总用时: {total_time/60:.1f} 分钟")
    print(f"成功: {completed - failed}, 失败: {failed}")
    print(f"结果保存在: {stats_dir}")


if __name__ == "__main__":
    main()
