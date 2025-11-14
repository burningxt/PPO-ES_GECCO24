#!/usr/bin/env python3
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python diff_plot.py <input.csv> <output.png>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Read CSV and extract data from the 'average' row
    with open(input_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # first line with episode names
        episodes = [int(h.replace("episode", "")) for h in header[1:]]

        values = None
        for row in reader:
            if row[0].strip().lower() == "average":
                values = [float(x) for x in row[1:]]
                break

    if values is None:
        print("Error: No 'average' row found in CSV.")
        sys.exit(1)

    # Compute val(x-1) - val(x); align with episode x (skip the first)
    diffs = np.array(values[:-1]) - np.array(values[1:])
    diff_episodes = episodes[1:]

    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(diff_episodes, diffs, marker='o', linestyle='-', color='tab:blue', label='Value(x-1) - Value(x)')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.title("Change vs Previous Episode: Value(x-1) - Value(x)", pad=20)
    plt.xlabel("Episode Number", labelpad=15)
    plt.ylabel("Value(x-1) - Value(x)", labelpad=15)

    # Show all episode labels
    plt.xticks(diff_episodes, rotation=45, ha='right')

    # Annotate each point with its numeric value
    for x, y in zip(diff_episodes, diffs):
        plt.text(
            x, y,
            f"{y:,.0f}",           # formatted with commas, no decimals
            ha='center', va='bottom',
            fontsize=9, rotation=30
        )

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Adjust margins for full visibility
    plt.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.25)

    # Linear y-axis with base-10 number formatting (no scientific notation)
    plt.yscale("linear")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')

    # Save with bounding box
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
