#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Plot the first subarray of a .npy file with x = evaluation index and y = value."
    )
    parser.add_argument("npy_file", help="Path to the .npy file")
    args = parser.parse_args()

    # Load data
    try:
        data = np.load(args.npy_file)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(1)

    # First array
    if data.ndim == 1:
        first = data
    else:
        first = data[0]

    x_vals = np.arange(len(first))
    y_vals = first

    print("Points (x = evaluation index, y = value):")
    for x, y in zip(x_vals, y_vals):
        print(f"{x}, {y}")

    # Larger figure for more horizontal and vertical space
    plt.figure(figsize=(14, 8))

    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel("Number of Generations")
    plt.ylabel("Solution Quality")
    plt.title("Best Found Solution of Each Generation")
    plt.grid(True)

    # Tilted labels at each point
    for x, y in zip(x_vals, y_vals):
        plt.text(x, y, f"{y}", fontsize=9, rotation=45, ha="left", va="bottom")

    plt.tight_layout()

    # Save PNG
    npy_path = Path(args.npy_file)
    output_png = npy_path.with_suffix(".png")
    plt.savefig(output_png)

    print(f"Saved plot to: {output_png}")

    plt.show()


if __name__ == "__main__":
    main()
