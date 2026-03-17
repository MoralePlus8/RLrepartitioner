#!/usr/bin/env python3
"""
从 stats/trace_combinations_100.csv 读取 4-trace 组合，
使用 bin/champsim_4core_even 模拟运行，
将统计数据输出到 stats/even_4core 目录。
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_BIN = PROJECT_ROOT / "bin" / "champsim_4core_even"
DEFAULT_TRACES_DIR = PROJECT_ROOT / "traces"
DEFAULT_COMBINATIONS_CSV = PROJECT_ROOT / "stats" / "trace_combinations_100.csv"
DEFAULT_STATS_DIR = PROJECT_ROOT / "stats" / "even_4core"
DEFAULT_WARMUP = 50_000_000       # 50M
DEFAULT_SIMULATION = 500_000_000  # 500M
DEFAULT_WORKERS = 8

TRACE_SUFFIXES = [".champsimtrace.xz", ".trace.xz"]


def resolve_trace_path(trace_name: str, traces_dir: Path) -> Path | None:
    """根据 trace 名称解析为 traces 目录下的完整路径"""
    for suffix in TRACE_SUFFIXES:
        path = traces_dir / f"{trace_name}{suffix}"
        if path.exists():
            return path
    return None


def load_combinations(csv_path: Path) -> list[tuple[int, str, str, str, str]]:
    """从 CSV 加载组合，返回 (combo_id, trace1, trace2, trace3, trace4) 列表"""
    try:
        import pandas as pd
    except ImportError:
        print("错误: 需要安装 pandas，请运行: pip install pandas", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = ["combo_id", "trace_1", "trace_2", "trace_3", "trace_4"]
    for col in required:
        if col not in df.columns:
            print(f"错误: CSV 缺少列 '{col}'", file=sys.stderr)
            sys.exit(1)

    return [
        (int(row["combo_id"]), row["trace_1"], row["trace_2"], row["trace_3"], row["trace_4"])
        for _, row in df.iterrows()
    ]


def run_single(args_tuple: tuple) -> dict:
    """运行单组四核模拟任务"""
    combo_id, trace_paths, trace_names, output_csv, champsim_bin, warmup, simulation = args_tuple

    result = {
        "combo_id": combo_id,
        "traces": "+".join(trace_names),
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
        *[str(p) for p in trace_paths],
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=86400,
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
        description="四核 Even 分区模拟：从 trace_combinations_100.csv 读取组合运行 ChampSim",
    )
    parser.add_argument(
        "--bin", type=str, default=str(DEFAULT_BIN),
        help=f"可执行文件路径 (默认: {DEFAULT_BIN})",
    )
    parser.add_argument(
        "--combinations-csv", type=str, default=str(DEFAULT_COMBINATIONS_CSV),
        help=f"组合列表 CSV 路径 (默认: {DEFAULT_COMBINATIONS_CSV})",
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
    combinations_csv = Path(args.combinations_csv)
    traces_dir = Path(args.traces_dir)
    stats_dir = Path(args.stats_dir)

    if not champsim_bin.exists():
        print(f"错误: 可执行文件不存在: {champsim_bin}")
        sys.exit(1)

    if not combinations_csv.exists():
        print(f"错误: 组合列表不存在: {combinations_csv}")
        sys.exit(1)

    if not traces_dir.exists():
        print(f"错误: Traces 目录不存在: {traces_dir}")
        sys.exit(1)

    raw_combos = load_combinations(combinations_csv)
    print(f"从 {combinations_csv} 加载 {len(raw_combos)} 个组合")

    # 解析 trace 路径，过滤无效组合
    tasks = []
    for combo_id, t1, t2, t3, t4 in raw_combos:
        paths = []
        for name in (t1, t2, t3, t4):
            p = resolve_trace_path(name, traces_dir)
            if p is None:
                print(f"警告: 跳过 combo {combo_id}，未找到 trace: {name}")
                break
            paths.append(p)
        if len(paths) == 4:
            output_name = f"{t1}+{t2}+{t3}+{t4}.csv"
            output_csv = str(stats_dir / output_name)
            tasks.append((combo_id, paths, (t1, t2, t3, t4), output_csv, str(champsim_bin),
                          args.warmup, args.simulation))

    if not tasks:
        print("错误: 没有有效的组合可运行")
        sys.exit(1)

    print(f"共有 {len(tasks)} 个有效组合")

    if args.skip_existing:
        before = len(tasks)
        tasks = [t for t in tasks if not os.path.exists(t[3])]
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
            print(f"  {i+1:4d}. combo {t[0]}: {' + '.join(t[2])}")
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
                f"combo {result['combo_id']}  ({result['duration']:.1f}s)  "
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
