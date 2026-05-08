#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "testdata"
VALUES_PER_LINE = 16


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))

    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def write_values(f, values: Iterable[float]) -> None:
    line: list[str] = []
    for value in values:
        line.append(format_value(value))
        if len(line) == VALUES_PER_LINE:
            f.write(" ".join(line) + "\n")
            line.clear()

    if line:
        f.write(" ".join(line) + "\n")


def write_case(path: Path, n: int, values: Iterable[float]) -> None:
    values_list = list(values)
    if len(values_list) != n:
        raise ValueError(f"Expected {n} values, got {len(values_list)}")

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        write_values(f, values_list)


def random_case(out_dir: Path, name: str, n: int, seed: int, lo: float = -12.0, hi: float = 12.0) -> None:
    rng = random.Random(seed)
    values = [rng.uniform(lo, hi) for _ in range(n)]
    write_case(out_dir / f"{name}.txt", n, values)


def repeating_pattern(n: int, pattern: list[float]) -> list[float]:
    return [pattern[i % len(pattern)] for i in range(n)]


def alternating_extremes(n: int) -> list[float]:
    return [80.0 if i % 2 == 0 else -80.0 for i in range(n)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate softmax fixed test data.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for generated .txt files (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_case(
        out_dir / "case_00_example_3.txt",
        3,
        [1.0, 2.0, 3.0],
    )

    write_case(
        out_dir / "case_01_example_5.txt",
        5,
        [-10.0, -5.0, 0.0, 5.0, 10.0],
    )

    write_case(
        out_dir / "case_02_single_1.txt",
        1,
        [42.0],
    )

    write_case(
        out_dir / "case_03_equal_32.txt",
        32,
        [7.0] * 32,
    )

    random_case(out_dir, "case_04_warp_31", 31, 101)
    random_case(out_dir, "case_05_block_boundary_256", 256, 202)
    random_case(out_dir, "case_06_block_misaligned_257", 257, 303)

    write_case(
        out_dir / "case_07_large_positive_64.txt",
        64,
        repeating_pattern(64, [1000.0, 999.0, 998.0, 997.0]),
    )

    write_case(
        out_dir / "case_08_large_negative_64.txt",
        64,
        repeating_pattern(64, [-1000.0, -999.0, -998.0, -997.0]),
    )

    write_case(
        out_dir / "case_09_stability_1024.txt",
        1024,
        alternating_extremes(1024),
    )

    random_case(out_dir, "case_10_perf_500000", 500_000, 404, lo=-20.0, hi=20.0)

    print("Generated 11 test files in", out_dir)


if __name__ == "__main__":
    main()
