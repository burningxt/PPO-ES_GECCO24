#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_csv(csv_path, output_path="plot.png"):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract episode labels (drop the first empty column)
    episodes = df.columns[1:]
    # Extract numeric values for the 'average' row
    values = df.iloc[0, 1:].astype(float)

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, values, marker='o', linestyle='-', linewidth=2)

    # Label each point with its value
    for i, val in enumerate(values):
        plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

    # Labels and title
    plt.title("Average (Across 25 different runs of every problem) Area Under Graph per Episode", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Area Under Graph", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add vertical spacing above the max value to prevent overlap
    y_min, y_max = values.min(), values.max()
    padding = (y_max - y_min) * 0.15  # add 15% vertical space
    plt.ylim(y_min - padding * 0.1, y_max + padding)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_csv.py <path_to_csv> [output_path]")
    else:
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "plot.png"
        plot_csv(csv_path, output_path)
