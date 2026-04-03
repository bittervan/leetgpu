#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "testdata"


def write_case(path: Path, width: int, height: int, rgba: list[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{width} {height}\n")
        for y in range(height):
            start = y * width * 4
            row = rgba[start : start + width * 4]
            f.write(" ".join(str(value) for value in row) + "\n")


def random_case(out_dir: Path, name: str, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed)
    rgba = [rng.randint(0, 255) for _ in range(width * height * 4)]
    write_case(out_dir / f"{name}.txt", width, height, rgba)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image_invert fixed test data.")
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

    for path in out_dir.glob("*.txt"):
        path.unlink()

    write_case(
        out_dir / "case_00_example_1x2.txt",
        1,
        2,
        [
            255,
            0,
            128,
            255,
            0,
            255,
            0,
            255,
        ],
    )

    write_case(
        out_dir / "case_01_example_2x1.txt",
        2,
        1,
        [
            10,
            20,
            30,
            255,
            100,
            150,
            200,
            255,
        ],
    )

    specs = [
        ("case_02_single_1x1", 1, 1, 101),
        ("case_03_row_7x1", 7, 1, 202),
        ("case_04_col_1x7", 1, 7, 303),
        ("case_05_rect_17x15_255_pixels", 17, 15, 404),
        ("case_06_square_16x16_256_pixels", 16, 16, 505),
        ("case_07_row_257x1_257_pixels", 257, 1, 606),
        ("case_08_rect_19x17_323_pixels", 19, 17, 707),
        ("case_09_large_65x33", 65, 33, 808),
    ]

    for name, width, height, seed in specs:
        random_case(out_dir, name, width, height, seed)

    print(f"Generated {len(specs) + 2} test files in {out_dir}")


if __name__ == "__main__":
    main()
