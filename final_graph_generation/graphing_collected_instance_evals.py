#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re

def _update_to_int(label: str) -> int:
    """
    Extract integer from labels like 'updates_3'
    """
    m = re.search(r"\d+", str(label))
    if not m:
        raise ValueError(f"Could not extract update number from {label!r}")
    return int(m.group())

def plot_csv(csv_path, output_path="collected_evals.png"):
    df = pd.read_csv(csv_path)

    # Extract update labels
    update_labels = list(df.columns[1:])
    updates = [_update_to_int(lbl) for lbl in update_labels]

    # Extract collected evals row
    values = df.iloc[0, 1:].astype(float).tolist()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(updates, values, marker='o', linewidth=2)

    # Label each point
    for x, y in zip(updates, values):
        plt.text(x, y, f"{y:.3f}",
                 ha='center', va='bottom',
                 fontsize=8)

    # Axis labels
    plt.xlabel("Number of Updates", fontsize=12)
    plt.ylabel("Collected Evals", fontsize=12)
    plt.title("Collected Evaluations vs Number of Updates", fontsize=14)

    # Grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Nice padding
    y_min, y_max = min(values), max(values)
    padding = (y_max - y_min) * 0.15 if y_max != y_min else 1.0
    plt.ylim(y_min - padding * 0.1, y_max + padding)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_collected_evals.py <path_to_csv> [output_path]")
    else:
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "collected_evals.png"
        plot_csv(csv_path, output_path)
