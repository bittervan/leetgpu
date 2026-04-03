#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "testdata"


def write_case(path: Path, rows: int, cols: int, values: list[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{rows} {cols}\n")
        for row in range(rows):
            start = row * cols
            f.write(" ".join(f"{value:.6f}" for value in values[start : start + cols]) + "\n")


def random_case(out_dir: Path, name: str, rows: int, cols: int, seed: int) -> None:
    rng = random.Random(seed)
    values = [rng.uniform(-10.0, 10.0) for _ in range(rows * cols)]
    write_case(out_dir / f"{name}.txt", rows, cols, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matrix_transpose fixed test data.")
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
        out_dir / "case_00_sanity_2x3.txt",
        2,
        3,
        [
            1.0,
            -2.0,
            3.5,
            0.0,
            4.0,
            -1.5,
        ],
    )

    specs = [
        ("case_01_single_1x1", 1, 1, 101),
        ("case_02_row_1x7", 1, 7, 202),
        ("case_03_col_7x1", 7, 1, 303),
        ("case_04_square_4x4", 4, 4, 404),
        ("case_05_rect_5x9", 5, 9, 505),
        ("case_06_tile_boundary_16x16", 16, 16, 606),
        ("case_07_tile_misaligned_17x19", 17, 19, 707),
        ("case_08_large_33x65", 33, 65, 808),
    ]

    for name, rows, cols, seed in specs:
        random_case(out_dir, name, rows, cols, seed)

    print(f"Generated {len(specs) + 1} test files in {out_dir}")


if __name__ == "__main__":
    main()
