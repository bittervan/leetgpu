#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "testdata"
VALUES_PER_LINE = 32


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
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        write_values(f, values)


def random_case(out_dir: Path, name: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    values = (float(rng.randint(-16, 16)) for _ in range(n))
    write_case(out_dir / f"{name}.txt", n, values)


def alternating_case(n: int) -> Iterable[float]:
    for index in range(n):
        yield 1.0 if index % 2 == 0 else -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reduction fixed test data.")
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
        out_dir / "case_00_example_8.txt",
        8,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )

    write_case(
        out_dir / "case_01_cancel_4.txt",
        4,
        [-2.5, 1.5, -1.0, 2.0],
    )

    write_case(
        out_dir / "case_02_single_1.txt",
        1,
        [42.0],
    )

    specs = [
        ("case_03_warp_31", 31, 101),
        ("case_04_warp_boundary_32", 32, 202),
        ("case_05_block_edge_255", 255, 303),
        ("case_06_block_boundary_256", 256, 404),
        ("case_07_block_misaligned_257", 257, 505),
        ("case_08_large_4096", 4096, 606),
    ]

    for name, n, seed in specs:
        random_case(out_dir, name, n, seed)

    write_case(
        out_dir / "case_09_perf_4194304.txt",
        4_194_304,
        alternating_case(4_194_304),
    )

    print(f"Generated {len(specs) + 4} test files in {out_dir}")


if __name__ == "__main__":
    main()
