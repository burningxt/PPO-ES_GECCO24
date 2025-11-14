#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot all problems on the same vivid-colored graph (value vs. episode)."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file")
    parser.add_argument("output_path", type=Path, help="Output image file path (e.g., all_problems.png)")
    parser.add_argument("--dpi", type=int, default=180, help="Image DPI")
    parser.add_argument("--yscale", type=str, default="linear", choices=["linear", "log", "symlog"], help="Y-axis scale")
    parser.add_argument("--title", type=str, default="All Problems Over Episodes", help="Plot title")
    parser.add_argument("--format", type=str, default="png", help="File format for saved image (png, pdf, svg)")
    parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=6.0, help="Figure height in inches")
    parser.add_argument("--legend_cols", type=int, default=None, help="Number of columns in legend (auto if not set)")
    return parser.parse_args()


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure first column is 'problem'
    if df.columns[0] is None or df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "problem"})
    else:
        df = df.rename(columns={df.columns[0]: "problem"})

    episode_cols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("episode")]
    for c in episode_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["problem", *episode_cols]]


def extract_episode_numbers(columns):
    episodes = []
    for c in columns:
        try:
            episodes.append(int(c.lower().replace("episode", "")))
        except Exception:
            episodes.append(None)
    return episodes


def main():
    args = parse_args()
    df = load_frame(args.csv_path)

    ep_cols = [c for c in df.columns if c != "problem"]
    x_vals = extract_episode_numbers(ep_cols)
    keep = [(c, x) for c, x in zip(ep_cols, x_vals) if x is not None]
    if not keep:
        raise ValueError("No valid episode columns found.")
    ep_cols, x_vals = zip(*keep)
    order = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
    ep_cols = [ep_cols[i] for i in order]
    x_vals = [x_vals[i] for i in order]

    num_problems = len(df)
    # Use a vivid color palette that supports many unique hues
    cmap = plt.cm.get_cmap("nipy_spectral", num_problems)
    colors = [cmap(i) for i in range(num_problems)]

    plt.figure(figsize=(args.width, args.height))

    for i, (_, row) in enumerate(df.iterrows()):
        problem = str(row["problem"])
        y = row[ep_cols].to_numpy(dtype=float)
        plt.plot(
            x_vals, y,
            label=problem,
            linewidth=1.8,
            color=colors[i],
            alpha=0.9
        )

    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title(args.title)
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.yscale(args.yscale)

    # Auto-determine legend columns if not specified
    legend_cols = args.legend_cols or max(2, int(np.ceil(np.sqrt(num_problems))))
    plt.legend(ncol=legend_cols, fontsize=8, loc="best", frameon=False)
    plt.tight_layout()

    out_path = args.output_path.with_suffix(f".{args.format}")
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()

    print(f"✅ Saved vivid combined plot with {num_problems} problems to {out_path.resolve()}")


if __name__ == "__main__":
    main()
