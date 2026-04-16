#!/usr/bin/env python3
"""
遍历目录下所有 .csv：仅根据文件末尾 4 行判断；
若其中任意一行的 instructions 列数值 < 5e8，则删除该文件。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 500_000_000


def should_delete_csv(path: Path, threshold: int) -> tuple[bool, str]:
    """
    返回 (是否应删除, 说明)。
    无法解析表头或缺少 instructions 列时不删除。
    """
    try:
        text_mode = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return False, f"读取失败: {e}"

    lines = text_mode.splitlines()
    if not lines:
        return False, "空文件"

    header_line = lines[0]
    try:
        header = next(csv.reader([header_line]))
    except Exception:
        return False, "表头解析失败"

    try:
        idx = header.index("instructions")
    except ValueError:
        return False, "无 instructions 列"

    # 文件最后 4 行（与「最后 4 行」字面一致；表头若落在这 4 行内会因无法转成数字而跳过）
    tail = lines[-4:] if len(lines) >= 4 else lines

    for line in tail:
        if not line.strip():
            continue
        try:
            row = next(csv.reader([line]))
        except Exception:
            continue
        if len(row) <= idx:
            continue
        raw = row[idx].strip()
        try:
            val = float(raw)
        except ValueError:
            continue
        if val < threshold:
            return True, f"instructions={raw} < {threshold}"

    return False, "保留"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "directory",
        type=Path,
        help="要扫描的目录（仅该目录内 *.csv，不含子目录）",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归包含子目录中的 .csv",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"instructions 下限（默认 {DEFAULT_THRESHOLD:g}）",
    )
    p.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="只打印将删除的文件，不真正删除",
    )
    args = p.parse_args()
    root: Path = args.directory.expanduser().resolve()
    if not root.is_dir():
        print(f"错误: 不是目录: {root}", file=sys.stderr)
        return 1

    pattern = "**/*.csv" if args.recursive else "*.csv"
    csv_files = sorted(root.glob(pattern))
    deleted = 0
    for path in csv_files:
        if not path.is_file():
            continue
        kill, reason = should_delete_csv(path, args.threshold)
        if not kill:
            continue
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        if args.dry_run:
            print(f"[dry-run] 将删除 {rel}: {reason}")
            deleted += 1
            continue
        try:
            path.unlink()
            print(f"已删除 {rel}: {reason}")
            deleted += 1
        except OSError as e:
            print(f"删除失败 {rel}: {e}", file=sys.stderr)

    print(f"完成: {'将删除' if args.dry_run else '已删除'} {deleted} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
