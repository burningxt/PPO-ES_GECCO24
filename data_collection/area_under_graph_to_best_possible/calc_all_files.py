#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

# Import the function from your file.
# Make sure average_auc.py is on your PYTHONPATH or in the same folder.
from average_area_under import average_auc

PATTERN = re.compile(
    r"fitness_episode_(?P<episode>\d+)_problem_(?P<problem>\d+)_instance_(?P<instance>\d+)\.npy$"
)

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate average AUC results into a problems x episodes CSV."
    )
    parser.add_argument("input_dir", help="Directory to scan for .npy files")
    parser.add_argument("output_csv", help="Path to write the output CSV")
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    paths = (
        root.rglob("*.npy") if args.recursive else root.glob("*.npy")
    )

    # Collect AUCs grouped by (problem, episode). Average over instances later.
    aucs = defaultdict(lambda: defaultdict(list))
    episodes_seen = set()
    problems_seen = set()

    for p in paths:
        m = PATTERN.match(p.name)
        if not m:
            continue
        ep = int(m.group("episode"))
        pr = int(m.group("problem"))
        # instance = int(m.group("instance"))  # not needed except for debugging

        try:
            val = average_auc(str(p))
        except Exception as e:
            print(f"Skip {p.name}: failed to compute AUC ({e})")
            continue

        aucs[pr][ep].append(val)
        episodes_seen.add(ep)
        problems_seen.add(pr)

    if not episodes_seen or not problems_seen:
        raise SystemExit("No matching files found with valid results.")

    # Sort episodes and problems numerically
    episodes = sorted(episodes_seen)
    problems = sorted(problems_seen)

    # Build header: ,episode1,episode60,...
    header = [""] + [f"episode{e}" for e in episodes]

    # Write CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pr in problems:
            row = [f"problem{pr}"]
            for ep in episodes:
                cell_vals = aucs[pr].get(ep, [])
                row.append(f"{mean(cell_vals):.6f}" if cell_vals else "")
            writer.writerow(row)

    print(f"Wrote CSV to {out_path.resolve()}")

if __name__ == "__main__":
    main()
