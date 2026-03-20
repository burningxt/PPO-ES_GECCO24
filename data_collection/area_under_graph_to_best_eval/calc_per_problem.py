#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from average_under_area_for_one_problem import average_auc

PATTERN = re.compile(
    r"fitness_episode_(?P<episode>\d+)_problem_(?P<problem>\d+)_instance_(?P<instance>\d+)\.npy$"
)

def main():
    parser = argparse.ArgumentParser(
        description="Plot mean AUC vs episode for each problem (mean over runs)."
    )
    parser.add_argument("input_dir", type=Path, help="Directory to scan for .npy files")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--dpi", type=int, default=160, help="Image DPI")
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "svg", "pdf"],
        help="Image format for plots",
    )
    parser.add_argument("--yscale", type=str, default="linear", choices=["linear", "log", "symlog"])
    parser.add_argument("--title_prefix", type=str, default="", help="Optional title prefix")
    parser.add_argument("--pdf", type=str, default=None, help="Optional combined PDF filename")
    args = parser.parse_args()

    root = args.input_dir.resolve()

    if not root.exists():
        raise SystemExit(f"Input directory does not exist: {root}")

    # Create output directory with same name + "_graphs"
    output_dir = root.parent / f"{root.name}_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = root.rglob("*.npy") if args.recursive else root.glob("*.npy")

    aucs = defaultdict(lambda: defaultdict(list))

    for p in paths:
        m = PATTERN.match(p.name)
        if not m:
            continue

        ep = int(m.group("episode"))
        pr = int(m.group("problem"))

        try:
            val = average_auc(str(p))
        except Exception as e:
            print(f"Skip {p.name}: failed to compute AUC ({e})")
            continue

        aucs[pr][ep].append(val)

    if not aucs:
        raise SystemExit("No matching files found with valid results.")

    pdf_path = output_dir / args.pdf if args.pdf else None
    pdf = PdfPages(pdf_path) if pdf_path else None

    num_plots = 0

    for pr in sorted(aucs.keys()):
        episodes = sorted(aucs[pr].keys())
        y_means = [mean(aucs[pr][ep]) for ep in episodes]

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(episodes, y_means, marker="o", linewidth=1.6, markersize=3.5)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Average area under the graph (mean across runs)")
        title = f"{args.title_prefix}problem{pr}" if args.title_prefix else f"problem{pr}"
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_yscale(args.yscale)

        fig.tight_layout()

        out_file = output_dir / f"problem{pr}.{args.format}"
        fig.savefig(out_file, dpi=args.dpi)

        if pdf:
            pdf.savefig(fig, dpi=args.dpi)

        plt.close(fig)
        num_plots += 1

    if pdf:
        pdf.close()
        print(f"Saved combined PDF: {pdf_path}")

    print(f"Saved {num_plots} plots to {output_dir.resolve()}")

if __name__ == "__main__":
    main()
