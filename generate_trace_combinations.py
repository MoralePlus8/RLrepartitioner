#!/usr/bin/env python3
"""
根据 stats/alone_apki_mpki.csv 中的 APKI 数据，生成 100 个 trace 组合。
每个组合包含 4 个 trace，且至少有一个 trace 的 APKI > 20。
"""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_APKI_CSV = PROJECT_ROOT / "stats" / "alone_apki_mpki.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "stats" / "trace_combinations_100.csv"


def load_apki_data(csv_path: Path) -> tuple[list[str], dict[str, float]]:
    """加载 APKI 数据，返回 trace 列表和 trace->apki 映射。"""
    try:
        import pandas as pd
    except ImportError:
        print("错误: 需要安装 pandas，请运行: pip install pandas", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    traces = df["trace"].tolist()
    apki_map = dict(zip(df["trace"], df["apki"]))
    return traces, apki_map


def has_high_apki_trace(combo: tuple[str, ...], apki_map: dict[str, float], threshold: float = 20.0) -> bool:
    """检查组合中是否至少有一个 trace 的 APKI 大于 threshold。"""
    return any(apki_map.get(t, 0) > threshold for t in combo)


def main():
    parser = argparse.ArgumentParser(
        description="生成满足 APKI 约束的 trace 组合列表",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_APKI_CSV,
        help=f"APKI 数据 CSV 路径 (默认: {DEFAULT_APKI_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出 CSV 路径 (默认: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="生成的组合数量 (默认: 100)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4,
        help="每个组合包含的 trace 数量 (默认: 4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="至少一个 trace 的 APKI 需大于此阈值 (默认: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于可复现结果",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    traces, apki_map = load_apki_data(input_path)
    n = len(traces)
    k = args.size

    if n < k:
        print(f"错误: trace 数量 ({n}) 少于组合大小 ({k})", file=sys.stderr)
        sys.exit(1)

    high_apki_traces = [t for t in traces if apki_map.get(t, 0) > args.threshold]
    if not high_apki_traces:
        print(f"错误: 没有 APKI > {args.threshold} 的 trace", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    seen: set[tuple[str, ...]] = set()
    combinations: list[tuple[str, ...]] = []

    max_attempts = 100_000
    attempts = 0

    while len(combinations) < args.count and attempts < max_attempts:
        attempts += 1
        combo = tuple(sorted(random.sample(traces, k)))
        if combo in seen:
            continue
        if not has_high_apki_trace(combo, apki_map, args.threshold):
            continue
        seen.add(combo)
        combinations.append(combo)

    if len(combinations) < args.count:
        print(
            f"警告: 仅生成 {len(combinations)} 个组合 (请求 {args.count})，"
            f"可能满足条件的组合已穷尽",
            file=sys.stderr,
        )

    # 写入 CSV
    try:
        import pandas as pd

        rows = []
        for i, combo in enumerate(combinations, start=1):
            row = {"combo_id": i}
            for j, trace in enumerate(combo, start=1):
                row[f"trace_{j}"] = trace
                row[f"trace_{j}_apki"] = round(apki_map[trace], 4)
            row["max_apki"] = round(max(apki_map[t] for t in combo), 4)
            row["has_high_apki"] = has_high_apki_trace(combo, apki_map, args.threshold)
            rows.append(row)

        df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"已生成 {len(combinations)} 个组合，结果已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 写入 CSV 失败 - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
