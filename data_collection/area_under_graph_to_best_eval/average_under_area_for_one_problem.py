import numpy as np
import sys

def average_auc(file_path):
    # Load data
    data = np.load(file_path, allow_pickle=True)

    # Check structure
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("Expected a list or array of lists in the .npy file.")

    areas = []

    # Iterate through each list
    for i, lst in enumerate(data):
        if not isinstance(lst, (list, np.ndarray)) or len(lst) == 0:
            print(f"Skipping list {i}: invalid or empty.")
            continue

        values = np.array(lst, dtype=float)
        # Shift curve so lowest value is at zero
        shifted = values - np.min(values)
        # Compute area under curve (trapezoidal rule, dx=1)
        area = np.trapz(shifted)
        areas.append(area)

    if not areas:
        raise ValueError("No valid lists found in the file.")

    avg_area = np.mean(areas)
    print(f"Average area under the curve across all lists: {avg_area}")
    return avg_area


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_npy_file>")
        sys.exit(1)
    average_auc(sys.argv[1])
