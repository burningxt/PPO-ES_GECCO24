#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args():
    parser = argparse.ArgumentParser(description="Plot value vs episodes for each problem from CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file")
    parser.add_argument("output_dir", type=Path, help="Directory to save plots")
    parser.add_argument("--pdf", type=str, default=None, help="Optional multi-page PDF filename (saved inside output_dir)")
    parser.add_argument("--dpi", type=int, default=160, help="Image DPI")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg", "jpeg", "svg", "pdf"], help="Image format for per-problem plots")
    parser.add_argument("--title_prefix", type=str, default="", help="Optional title prefix for each chart")
    parser.add_argument("--yscale", type=str, default="linear", choices=["linear", "log", "symlog"], help="Y scale for plots")
    return parser.parse_args()


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / args.pdf if args.pdf else None
    pdf = PdfPages(pdf_path) if pdf_path else None

    for _, row in df.iterrows():
        problem = str(row["problem"])
        y = row[ep_cols].to_numpy(dtype=float)

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(x_vals, y, marker="o", linewidth=1.6, markersize=3.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.set_title(f"{args.title_prefix}{problem}" if args.title_prefix else problem)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_yscale(args.yscale)
        fig.tight_layout()

        out_file = args.output_dir / f"{problem.replace('/', '_').replace(' ', '_')}.{args.format}"
        fig.savefig(out_file, dpi=args.dpi)
        if pdf:
            pdf.savefig(fig, dpi=args.dpi)
        plt.close(fig)

    if pdf:
        pdf.close()
        print(f"Saved combined PDF: {pdf_path}")

    print(f"Saved {len(df)} plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
