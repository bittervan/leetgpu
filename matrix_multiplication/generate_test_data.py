#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "testdata"


def write_case(path: Path, M: int, N: int, K: int, A: list[float], B: list[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{M} {N} {K}\n")
        for row in range(M):
            start = row * N
            values = " ".join(f"{value:.6f}" for value in A[start : start + N])
            f.write(values + "\n")
        f.write("\n")
        for row in range(N):
            start = row * K
            values = " ".join(f"{value:.6f}" for value in B[start : start + K])
            f.write(values + "\n")


def random_case(out_dir: Path, name: str, M: int, N: int, K: int, seed: int) -> None:
    rng = random.Random(seed)
    A = [rng.uniform(-3.0, 3.0) for _ in range(M * N)]
    B = [rng.uniform(-3.0, 3.0) for _ in range(N * K)]
    write_case(out_dir / f"{name}.txt", M, N, K, A, B)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matrix_multiplication fixed test data.")
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
        out_dir / "case_00_sanity_2x3x4.txt",
        2,
        3,
        4,
        [
            1.0,
            -2.0,
            3.0,
            0.5,
            -1.0,
            4.0,
        ],
        [
            2.0,
            -1.0,
            0.0,
            3.0,
            1.0,
            -2.0,
            -1.0,
            2.0,
            1.0,
            4.0,
            -3.0,
            0.5,
        ],
    )

    specs = [
        ("case_01_square_4x4x4", 4, 4, 4, 101),
        ("case_02_rect_5x7x3", 5, 7, 3, 202),
        ("case_03_odd_17x19x13", 17, 19, 13, 303),
        ("case_04_tall_31x9x27", 31, 9, 27, 404),
        ("case_05_wide_8x29x11", 8, 29, 11, 505),
        ("case_06_block_edge_16x16x16", 16, 16, 16, 606),
        ("case_07_block_misaligned_33x35x17", 33, 35, 17, 707),
    ]

    for name, M, N, K, seed in specs:
        random_case(out_dir, name, M, N, K, seed)

    print(f"Generated {len(specs) + 1} test files in {out_dir}")


if __name__ == "__main__":
    main()
