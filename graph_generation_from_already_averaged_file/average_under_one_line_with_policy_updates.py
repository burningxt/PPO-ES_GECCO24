#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re

def _episode_to_int(label: str) -> int:
    m = re.search(r"\d+", str(label))
    if not m:
        raise ValueError(f"Could not extract episode number from {label!r}")
    return int(m.group())

def plot_csv(csv_path, output_path="plot.png"):
    df = pd.read_csv(csv_path)

    # Original string labels
    episode_labels = list(df.columns[1:])
    values = df.iloc[0, 1:].astype(float).tolist()

    # Convert to numeric episode values
    episodes = [_episode_to_int(lbl) for lbl in episode_labels]

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, values, marker='o', linestyle='-', linewidth=2)

    # Annotate points
    for ep, val in zip(episodes, values):
        plt.text(ep, val, f"{val:.2f}",
                 ha='center', va='bottom',
                 fontsize=8, rotation=45)

    # ---- Vertical red dotted lines every 120 episodes ----
    min_ep = min(episodes)
    max_ep = max(episodes)

    for ep in range((min_ep // 120) * 120, max_ep + 120, 120):
        plt.axvline(x=ep, color='red', linestyle=':', linewidth=1)

    # ---- Keep ticks every 60 episodes ----
    tick_positions = []
    tick_labels = []

    for ep, lbl in zip(episodes, episode_labels):
        if ep % 60 == 0:
            tick_positions.append(ep)
            tick_labels.append(lbl)

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title("Average (Across 25 different runs of every problem) Area Under Graph per Episode", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Area Under Graph", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.6)

    # Padding
    y_min, y_max = min(values), max(values)
    padding = (y_max - y_min) * 0.15 if y_max != y_min else 1.0
    plt.ylim(y_min - padding * 0.1, y_max + padding)

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
