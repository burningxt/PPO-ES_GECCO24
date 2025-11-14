#!/usr/bin/env python3
import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_file.npy>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Load the NumPy file
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        sys.exit(1)

    # Ensure the file contains a list of lists
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("Expected a list or NumPy array in the file")

    final_values = []

    # Print the final value of each list and collect them
    for i, lst in enumerate(data):
        if isinstance(lst, (list, np.ndarray)) and len(lst) > 0:
            final_value = float(lst[-1])
            final_values.append(final_value)
            print(f"List {i} final value:", final_value)
        else:
            print(f"List {i} is empty or not a list-like object")

    # Calculate and print the average if there are valid values
    if final_values:
        average_value = np.mean(final_values)
        print(f"\nAverage of final values: {average_value}")
    else:
        print("\nNo valid lists found to calculate average.")

if __name__ == "__main__":
    main()
