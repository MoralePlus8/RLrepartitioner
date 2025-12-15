#!/usr/bin/env python3
"""
批量运行ChampSim模拟器，遍历所有trace文件的两两组合
使用多进程并行运行以提高效率
"""

import os
import sys
import glob
import subprocess
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import time

# 配置参数
PROJECT_ROOT = Path(__file__).parent.resolve()
CHAMPSIM_BIN = PROJECT_ROOT / "bin" / "champsim"
DEFAULT_TRACES_DIR = PROJECT_ROOT / "traces"
DEFAULT_STATS_DIR = PROJECT_ROOT / "stats"
DEFAULT_WARMUP = 200000000
DEFAULT_SIMULATION = 500000000
DEFAULT_WORKERS = 8


def get_trace_files(traces_dir: Path) -> list:
    """获取traces目录下所有的trace文件"""
    # 支持 .champsimtrace.xz 和 .trace.xz 格式
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
    # 移除各种可能的后缀
    for suffix in [".champsimtrace.xz", ".trace.xz"]:
        if basename.endswith(suffix):
            return basename[:-len(suffix)]
    return basename


def run_simulation(args_tuple: tuple) -> dict:
    """
    运行单个模拟任务
    
    Args:
        args_tuple: (trace1_path, trace2_path, output_csv_path, champsim_bin, warmup, simulation) 元组
    
    Returns:
        包含任务结果信息的字典
    """
    trace1, trace2, output_csv, champsim_bin, warmup, simulation = args_tuple
    trace1_name = get_trace_name(trace1)
    trace2_name = get_trace_name(trace2)
    
    result = {
        "trace1": trace1_name,
        "trace2": trace2_name,
        "output": output_csv,
        "success": False,
        "error": None,
        "duration": 0
    }
    
    start_time = time.time()
    
    # 构建命令，使用 --csv-output 参数指定输出路径
    cmd = [
        str(champsim_bin),
        "--warmup-instructions", str(warmup),
        "--simulation-instructions", str(simulation),
        "--csv-output", output_csv,
        trace1,
        trace2
    ]
    
    try:
        # 运行champsim
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=36000  # 10小时超时
        )
        
        # 检查是否成功生成了CSV文件
        if os.path.exists(output_csv):
            result["success"] = True
        else:
            result["error"] = f"CSV file not generated. Return code: {process.returncode}"
            if process.stderr:
                result["error"] += f"\nStderr: {process.stderr[:500]}"
                
    except subprocess.TimeoutExpired:
        result["error"] = "Simulation timed out (>10 hours)"
    except Exception as e:
        result["error"] = str(e)
    
    result["duration"] = time.time() - start_time
    return result


def generate_task_list(traces: list, stats_dir: Path, champsim_bin: Path, 
                       warmup: int, simulation: int) -> list:
    """
    生成所有任务列表
    
    Args:
        traces: trace文件路径列表
        stats_dir: 统计文件输出目录
        champsim_bin: champsim可执行文件路径
        warmup: 预热指令数
        simulation: 模拟指令数
    
    Returns:
        任务元组列表
    """
    tasks = []
    
    # 生成所有两两组合
    for trace1, trace2 in itertools.combinations(traces, 2):
        name1 = get_trace_name(trace1)
        name2 = get_trace_name(trace2)
        output_csv = str(stats_dir / f"{name1}+{name2}.csv")
        tasks.append((trace1, trace2, output_csv, str(champsim_bin), warmup, simulation))
    
    return tasks


def filter_existing_tasks(tasks: list) -> list:
    """过滤掉已经存在结果的任务"""
    filtered = []
    for task in tasks:
        output_csv = task[2]
        if not os.path.exists(output_csv):
            filtered.append(task)
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="批量运行ChampSim模拟器，遍历所有trace文件的两两组合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all_combinations.py --dry-run                    # 预览任务
  python run_all_combinations.py --limit 10                   # 只运行10个任务（测试）
  python run_all_combinations.py --skip-existing              # 跳过已有结果
  python run_all_combinations.py --workers 4                  # 使用4个并行进程
        """
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
        default=str(DEFAULT_STATS_DIR),
        help=f"统计结果输出目录 (默认: {DEFAULT_STATS_DIR})"
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
    
    traces_dir = Path(args.traces_dir)
    stats_dir = Path(args.stats_dir)
    
    # 检查champsim是否存在
    if not CHAMPSIM_BIN.exists():
        print(f"错误: ChampSim可执行文件不存在: {CHAMPSIM_BIN}")
        print("请先编译ChampSim: ./config.sh && make")
        sys.exit(1)
    
    # 检查traces目录
    if not traces_dir.exists():
        print(f"错误: Traces目录不存在: {traces_dir}")
        sys.exit(1)
    
    # 获取所有trace文件
    traces = get_trace_files(traces_dir)
    if not traces:
        print(f"错误: 在 {traces_dir} 目录下未找到trace文件")
        print("支持的格式: *.champsimtrace.xz, *.trace.xz")
        sys.exit(1)
    
    print(f"找到 {len(traces)} 个trace文件")
    
    # 生成任务列表
    tasks = generate_task_list(traces, stats_dir, CHAMPSIM_BIN, 
                               args.warmup, args.simulation)
    total_combinations = len(tasks)
    print(f"共有 {total_combinations} 个组合 (C({len(traces)},2) = {len(traces)}*{len(traces)-1}/2)")
    
    # 过滤已存在的任务
    if args.skip_existing:
        tasks = filter_existing_tasks(tasks)
        skipped = total_combinations - len(tasks)
        if skipped > 0:
            print(f"跳过 {skipped} 个已存在的结果文件")
    
    # 限制任务数量
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"限制运行 {len(tasks)} 个任务")
    
    if not tasks:
        print("没有任务需要运行")
        return
    
    # 创建输出目录
    os.makedirs(stats_dir, exist_ok=True)
    
    # Dry run模式
    if args.dry_run:
        print(f"\n将要运行的任务 (共 {len(tasks)} 个):")
        print("-" * 60)
        for i, task in enumerate(tasks[:20]):
            trace1, trace2 = task[0], task[1]
            print(f"  {i+1:4d}. {get_trace_name(trace1)} + {get_trace_name(trace2)}")
        if len(tasks) > 20:
            print(f"  ... 还有 {len(tasks) - 20} 个任务")
        print("-" * 60)
        print(f"输出目录: {stats_dir}")
        print(f"使用 --workers {args.workers} 个并行进程")
        return
    
    # 运行模拟
    print(f"\n使用 {args.workers} 个进程并行运行 {len(tasks)} 个模拟任务...")
    print("=" * 70)
    
    start_time = time.time()
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(run_simulation, task): task for task in tasks}
        
        # 处理完成的任务
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
            remaining = (len(tasks) - completed) * avg_time / args.workers
            
            print(f"[{completed:4d}/{len(tasks)}] {status} {result['trace1']} + {result['trace2']} "
                  f"({result['duration']:.1f}s) - 预计剩余: {remaining/60:.1f}分钟")
            
            if not result["success"]:
                print(f"         错误: {result['error']}")
    
    # 输出统计
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"完成！总用时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"成功: {completed - failed}, 失败: {failed}")
    print(f"结果保存在: {stats_dir}")


if __name__ == "__main__":
    main()

