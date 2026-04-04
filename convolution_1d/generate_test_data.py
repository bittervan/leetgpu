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


def write_case(path: Path, input_values: list[float], kernel_values: list[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(input_values)} {len(kernel_values)}\n")
        write_values(f, input_values)
        f.write("\n")
        write_values(f, kernel_values)


def random_case(out_dir: Path, name: str, input_size: int, kernel_size: int, seed: int) -> None:
    rng = random.Random(seed)
    input_values = [rng.uniform(-2.0, 2.0) for _ in range(input_size)]
    kernel_values = [rng.uniform(-2.0, 2.0) for _ in range(kernel_size)]
    write_case(out_dir / f"{name}.txt", input_values, kernel_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate convolution_1d fixed test data.")
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
        out_dir / "case_00_sanity_5_3.txt",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 0.0, -1.0],
    )

    specs = [
        ("case_01_single_1_1", 1, 1, 101),
        ("case_02_full_overlap_7_7", 7, 7, 202),
        ("case_03_output_31", 35, 5, 303),
        ("case_04_output_32", 36, 5, 404),
        ("case_05_output_255", 263, 9, 505),
        ("case_06_output_256", 264, 9, 606),
        ("case_07_output_257", 265, 9, 707),
        ("case_08_large_4096_31", 4096, 31, 808),
    ]

    for name, input_size, kernel_size, seed in specs:
        random_case(out_dir, name, input_size, kernel_size, seed)

    print(f"Generated {len(specs) + 1} test files in {out_dir}")


if __name__ == "__main__":
    main()
