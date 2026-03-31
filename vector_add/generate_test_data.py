#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "testdata"
VALUES_PER_LINE = 16


def write_values(f, values: list[float]) -> None:
    for start in range(0, len(values), VALUES_PER_LINE):
        chunk = values[start : start + VALUES_PER_LINE]
        f.write(" ".join(f"{value:.6f}" for value in chunk) + "\n")


def write_case(path: Path, n: int, a: list[float], b: list[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        write_values(f, a)
        f.write("\n")
        write_values(f, b)


def random_case(out_dir: Path, name: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    a = [rng.uniform(-10.0, 10.0) for _ in range(n)]
    b = [rng.uniform(-10.0, 10.0) for _ in range(n)]
    write_case(out_dir / f"{name}.txt", n, a, b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate vector_add fixed test data.")
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
        out_dir / "case_00_sanity_4.txt",
        4,
        [1.0, -2.0, 3.5, 0.0],
        [2.0, 4.0, -1.5, 7.0],
    )

    specs = [
        ("case_01_single_1", 1, 101),
        ("case_02_small_7", 7, 202),
        ("case_03_warp_31", 31, 303),
        ("case_04_warp_boundary_32", 32, 404),
        ("case_05_block_edge_255", 255, 505),
        ("case_06_block_boundary_256", 256, 606),
        ("case_07_block_misaligned_257", 257, 707),
        ("case_08_large_1024", 1024, 808),
    ]

    for name, n, seed in specs:
        random_case(out_dir, name, n, seed)

    print(f"Generated {len(specs) + 1} test files in {out_dir}")


if __name__ == "__main__":
    main()
