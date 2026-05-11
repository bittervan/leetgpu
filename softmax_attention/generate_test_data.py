#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "testdata"


def fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def write_matrix(f, values: list[float], rows: int, cols: int) -> None:
    for row in range(rows):
        start = row * cols
        f.write(" ".join(fmt(v) for v in values[start : start + cols]) + "\n")


def write_case(path: Path, M: int, N: int, d: int, Q: list[float], K: list[float], V: list[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{M} {N} {d}\n")
        write_matrix(f, Q, M, d)
        f.write("\n")
        write_matrix(f, K, N, d)
        f.write("\n")
        write_matrix(f, V, N, d)


def random_case(out_dir: Path, name: str, M: int, N: int, d: int, seed: int) -> None:
    rng = random.Random(seed)
    Q = [rng.uniform(-2.0, 2.0) for _ in range(M * d)]
    K = [rng.uniform(-2.0, 2.0) for _ in range(N * d)]
    V = [rng.uniform(-2.0, 2.0) for _ in range(N * d)]
    write_case(out_dir / f"{name}.txt", M, N, d, Q, K, V)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate softmax attention fixed test data.")
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
        out_dir / "case_00_example_2x3x4.txt",
        2,
        3,
        4,
        [
            1.0,
            -2.0,
            0.5,
            1.5,
            -1.0,
            2.0,
            3.0,
            -0.5,
        ],
        [
            0.5,
            1.0,
            -1.5,
            2.0,
            1.0,
            -0.5,
            0.0,
            1.5,
            -1.0,
            0.5,
            2.0,
            -2.0,
        ],
        [
            1.0,
            0.0,
            0.5,
            -1.0,
            -0.5,
            1.5,
            2.0,
            0.0,
            0.5,
            -1.5,
            1.0,
            2.0,
        ],
    )

    write_case(
        out_dir / "case_01_single_1x1x1.txt",
        1,
        1,
        1,
        [2.0],
        [3.0],
        [4.0],
    )

    specs = [
        ("case_02_warp_m_31_n_32_d_8", 31, 32, 8, 101),
        ("case_03_warp_n_32_m_32_d_16", 32, 32, 16, 202),
        ("case_04_block_boundary_32x32x32", 32, 32, 32, 303),
        ("case_05_block_misaligned_33x35x17", 33, 35, 17, 404),
    ]

    for name, M, N, d, seed in specs:
        random_case(out_dir, name, M, N, d, seed)

    random_case(out_dir, "case_06_perf_512x256x128", 512, 256, 128, 505)

    print(f"Generated {len(specs) + 3} test files in {out_dir}")


if __name__ == "__main__":
    main()
