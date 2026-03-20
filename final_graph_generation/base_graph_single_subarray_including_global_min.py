#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..",  "bbob_optima.json"))

def get_f_opt(dim, instance, problem_index):
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    dim_key = f"dim_{dim}"
    inst_key = f"instance_{instance}"
    func_key = f"f{problem_index:02d}"

    return data[dim_key][inst_key][func_key]["f_opt"]

def parse_filename(file_path):
    filename = os.path.basename(file_path)
    pattern = r"fitness_episode_(\d+)_problem_(\d+)_instance_(\d+)\.npy"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError("Filename does not match expected pattern.")

    episode, problem, instance = map(int, match.groups())
    return episode, problem, instance

def get_dimension_from_path(file_path):
    parent = os.path.basename(os.path.dirname(file_path))
    match = re.match(r"DIM_(\d+)", parent)
    if not match:
        raise ValueError("Parent directory does not match expected DIM_<dim> pattern.")
    return int(match.group(1))

def main():
    parser = argparse.ArgumentParser(
        description="Plot the first subarray of a .npy file and draw global optimum line with shaded area."
    )
    parser.add_argument("npy_file", help="Path to the .npy file")
    args = parser.parse_args()

    npy_path = Path(args.npy_file)

    # Load npy
    data = np.load(npy_path, allow_pickle=True)
    first = data if data.ndim == 1 else data[0]
    y_vals = np.array(first, dtype=float)
    x_vals = np.arange(len(y_vals))

    # Parse metadata
    dim = get_dimension_from_path(str(npy_path))
    episode, problem, instance = parse_filename(str(npy_path))

    # Get f_opt
    f_opt = get_f_opt(dim, instance, problem)

    # Plot setup
    plt.figure(figsize=(14, 8))

    plt.plot(x_vals, y_vals, marker="o", label="Best found solution")

    # Global minimum line
    plt.axhline(y=f_opt, linestyle="--", color="red")

    # Label under the line
    plt.text(
        x_vals[-1],
        f_opt - 0.02 * (max(y_vals) - min(y_vals)),
        f"Global minimum ({f_opt})",
        fontsize=12,
        ha="right",
        va="top",
        color="red"
    )

    # Shade between curve and minimum
    plt.fill_between(x_vals, y_vals, f_opt, where=(y_vals >= f_opt), alpha=0.3)

    plt.xlabel("Number of Generations")
    plt.ylabel("Solution Quality")


    plt.title(
        f"Best Found Solution of Each Generation\n"
        f"(Problem: {problem}, Episodes: {episode}, DIM: {dim}, Instance: {instance})",
        # f"(dim = {dim}, problem = {problem}, instance = {instance}, episode = {episode})",
        fontsize=14
    )

    plt.grid(True)

    # Point labels
    for x, y in zip(x_vals, y_vals):
        plt.text(x, y, f"{y}", fontsize=9, rotation=45, ha="left", va="bottom")

    plt.tight_layout()

# Save image using the exact same filename as the numpy file
    base_name = npy_path.stem              # removes .npy
    output_png = npy_path.parent / f"{base_name}.png"

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_png}")


    # plt.show()

if __name__ == "__main__":
    main()
