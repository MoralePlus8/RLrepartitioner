#!/usr/bin/env python3
"""删除指定目录下 instructions 不足 5 亿的 CSV 文件。

对每个 CSV 只检查最后 4 行数据；若任一行 instructions < 500000000，则删除该文件。
"""

import argparse
import csv
from pathlib import Path


def should_delete(csv_path: Path, threshold: int = 500_000_000) -> bool:
    """检查文件最后 4 行是否有 instructions < threshold。"""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return False

    if not rows:
        return False

    last_4 = rows[-4:]
    for row in last_4:
        try:
            val = int(row.get("instructions", 0))
            if val < threshold:
                return True
        except (ValueError, TypeError):
            pass
    return False


def main():
    parser = argparse.ArgumentParser(description="按 instructions 阈值筛选并删除 CSV")
    parser.add_argument("dir", type=Path, help="待扫描的目录")
    parser.add_argument(
        "--threshold",
        type=int,
        default=500_000_000,
        help="instructions 阈值（默认 500000000）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅列出将被删除的文件，不实际删除",
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"错误：{args.dir} 不是目录")
        return 1

    deleted = []
    for csv_path in sorted(args.dir.glob("*.csv")):
        if should_delete(csv_path, args.threshold):
            deleted.append(csv_path)

    if not deleted:
        print(f"未发现需要删除的 CSV（阈值={args.threshold:,}）")
        return 0

    print(f"将删除以下 {len(deleted)} 个文件（instructions < {args.threshold:,}）：")
    for p in deleted:
        print(f"  {p}")

    if args.dry_run:
        print("\n[--dry-run] 未执行删除")
        return 0

    for p in deleted:
        p.unlink()
        print(f"已删除: {p}")
    return 0


if __name__ == "__main__":
    exit(main())
