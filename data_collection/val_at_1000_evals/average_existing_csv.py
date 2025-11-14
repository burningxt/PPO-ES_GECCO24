#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python avg_episodes.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load CSV with problems as index
    df = pd.read_csv(input_path, index_col=0)

    # Convert to numeric (ignore non-numeric cells)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Compute mean across problems for each episode
    column_means = df.mean(axis=0, skipna=True)

    # Build result DataFrame
    result_df = pd.DataFrame([column_means.values], columns=column_means.index, index=["average"])

    # Save to CSV
    result_df.to_csv(output_path)

    print(f"Averages saved to: {output_path}")
    print(result_df.to_csv(index=True), end="")

if __name__ == "__main__":
    main()
