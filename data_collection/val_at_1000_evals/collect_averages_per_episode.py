#!/usr/bin/env python3
import sys
import numpy as np
import csv
import re
from pathlib import Path
from collections import defaultdict

# Match filenames like fitness_episode_1_problem_3_instance_2.npy
PATTERN = re.compile(
    r"fitness_episode_(?P<episode>\d+)_problem_(?P<problem>\d+)_instance_(?P<instance>\d+)\.npy$"
)

def load_file_mean(path: Path):
    """Return mean of the final values from all lists inside .npy."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Skip {path.name}: failed to load ({e})")
        return None

    finals = []
    try:
        for row in data:
            if isinstance(row, (list, np.ndarray)) and len(row) > 0:
                finals.append(float(row[-1]))
    except Exception as e:
        print(f"Skip {path.name}: invalid structure ({e})")
        return None

    if finals:
        return float(np.mean(finals))
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <directory>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.exists() or not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory")
        sys.exit(1)

    cell_values = defaultdict(list)
    episodes = set()
    problems = set()

    # Collect means per (problem, episode)
    for path in folder.rglob("*.npy"):
        m = PATTERN.match(path.name)
        if not m:
            continue

        episode = int(m.group("episode"))
        problem = int(m.group("problem"))
        episodes.add(episode)
        problems.add(problem)

        mean_val = load_file_mean(path)
        if mean_val is not None:
            cell_values[(problem, episode)].append(mean_val)
        else:
            print(f"Note: {path.name} had no valid final entries")

    # Sort columns and rows
    episodes = sorted(episodes)
    problems = sorted(problems)

    header = [""] + [f"episode{e}" for e in episodes]
    rows = [header]

    # Build CSV table
    for p in problems:
        row = [f"problem{p}"]
        for e in episodes:
            vals = cell_values.get((p, e), [])
            if vals:
                row.append(f"{np.mean(vals):.6f}")
            else:
                row.append("")
        rows.append(row)

    # Write results
    with open("fitness_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("Saved fitness_matrix.csv")

if __name__ == "__main__":
    main()
