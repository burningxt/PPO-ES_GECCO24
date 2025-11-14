#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Average across problems with 95% CI. Auto y-limits; auto log/symlog if negatives exist."
    )
    p.add_argument("csv_path", type=Path, help="Path to the CSV file")
    p.add_argument("output_path", type=Path, help="Output image file path, e.g., avg_ci.png")
    p.add_argument("--dpi", type=int, default=180, help="Image DPI")
    p.add_argument(
        "--yscale",
        type=str,
        default="auto",
        choices=["auto", "linear", "log", "symlog"],
        help="Axis scaling. auto picks log if all positive else symlog",
    )
    p.add_argument("--title", type=str, default="Average Across Problems with 95% CI", help="Plot title")
    p.add_argument("--format", type=str, default="png", help="Saved image format: png, pdf, svg")
    p.add_argument("--width", type=float, default=10.0, help="Figure width in inches")
    p.add_argument("--height", type=float, default=6.0, help="Figure height in inches")
    p.add_argument("--color", type=str, default="crimson", help="Line color")
    p.add_argument("--ci_alpha", type=float, default=0.25, help="Alpha for CI band")
    # symlog tuning
    p.add_argument("--linthresh", type=float, default=1.0, help="symlog linear threshold")
    p.add_argument("--linscale", type=float, default=1.0, help="symlog linear scale")
    return p.parse_args()


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "problem"})
    ep_cols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("episode")]
    for c in ep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["problem", *ep_cols]]


def extract_episode_numbers(columns):
    eps = []
    for c in columns:
        try:
            eps.append(int(c.lower().replace("episode", "")))
        except Exception:
            eps.append(None)
    return eps


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
    x_vals = np.array([x_vals[i] for i in order], dtype=float)

    # Stats across problems per episode
    vals = df[ep_cols]
    y_mean = vals.mean(axis=0).to_numpy(dtype=float)
    y_std  = vals.std(axis=0, ddof=1).to_numpy(dtype=float)
    n      = vals.count(axis=0).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_sem = np.where(n > 0, y_std / np.sqrt(n), np.nan)
    ci_half = 1.96 * y_sem
    y_low = y_mean - ci_half
    y_high = y_mean + ci_half

    # Determine scale automatically if requested
    has_nonpos = np.nanmin(np.concatenate([y_mean, y_low, y_high])) <= 0
    scale = args.yscale
    if scale == "auto":
        scale = "symlog" if has_nonpos else "log"

    # For pure log, mask non-positive to avoid errors
    if scale == "log":
        mask = (y_mean > 0) & (y_low > 0) & (y_high > 0)
        y_mean_plot = np.where(mask, y_mean, np.nan)
        y_low_plot  = np.where(mask, y_low,  np.nan)
        y_high_plot = np.where(mask, y_high, np.nan)
    else:
        y_mean_plot, y_low_plot, y_high_plot = y_mean, y_low, y_high

    # Compute automatic y-limits from data (bottom from true minimum)
    all_y = np.concatenate([y_low_plot, y_mean_plot, y_high_plot])
    finite = np.isfinite(all_y)
    if not np.any(finite):
        raise ValueError("All y values are non-finite after processing.")
    y_min_data = float(np.nanmin(all_y[finite]))
    y_max_data = float(np.nanmax(all_y[finite]))

    # Pad limits a bit
    if scale == "log":
        # Both limits must be positive
        # For the bottom, use the smallest positive value from data
        pos_vals = all_y[finite & (all_y > 0)]
        if len(pos_vals) == 0:
            raise ValueError("No positive values available for log scale.")
        ymin = float(np.nanmin(pos_vals)) * 0.9
        ymax = float(np.nanmax(pos_vals)) * 1.2
    else:
        # linear or symlog can include negatives
        span = y_max_data - y_min_data
        pad = 0.05 * span if span > 0 else 1.0
        ymin = y_min_data - pad
        ymax = y_max_data + pad

    plt.figure(figsize=(args.width, args.height))
    plt.fill_between(x_vals, y_low_plot, y_high_plot, color=args.color, alpha=args.ci_alpha, label="95% CI")
    plt.plot(x_vals, y_mean_plot, color=args.color, linewidth=2.6, marker="o", markersize=5, label="Average")

    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title(args.title)
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    # Apply scale
    if scale == "symlog":
        plt.yscale("symlog", linthresh=args.linthresh, linscale=args.linscale)
    else:
        plt.yscale(scale)

    # Axis limits
    plt.xlim(np.nanmin(x_vals) * 0.95, np.nanmax(x_vals) * 1.05)
    plt.ylim(ymin, ymax)

    plt.legend(fontsize=9, loc="best", frameon=False)
    plt.tight_layout()

    out_path = args.output_path.with_suffix(f".{args.format}")
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()

    print(f"Saved to {out_path.resolve()}")
    print(f"Scale: {scale} | y-limits: [{ymin:.6g}, {ymax:.6g}] | x-limits: [{x_vals.min():.6g}, {x_vals.max():.6g}]")


if __name__ == "__main__":
    main()
