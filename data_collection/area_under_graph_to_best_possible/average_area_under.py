import numpy as np
import sys
import re
import os
import json 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "bbob_optima.json"))

def get_f_opt(dim, instance, problem_index):
    """Load JSON and return f_opt for dim, instance, and problem number."""
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    dim_key = f"dim_{dim}"
    inst_key = f"instance_{instance}"
    func_key = f"f{problem_index:02d}"

    try:
        return data[dim_key][inst_key][func_key]["f_opt"]
    except KeyError as e:
        raise KeyError(
            f"Missing key {e} while searching for: "
            f"{dim_key} → {inst_key} → {func_key}"
        )
    
def average_auc(file_path):

    parent = os.path.basename(os.path.dirname(file_path))
    match = re.match(r"DIM_(\d+)", parent)
    dim = int(match.group(1))

    episode, problem, instance = parse_filename(file_path)

    print(f"Parsed from filename:")
    print(f"  Episode:  {episode}")
    print(f"  Problem:  {problem}")
    print(f"  Instance: {instance}")
    print(f"  Dim: {dim}")

    f_opt = get_f_opt(dim, instance, problem)
    print(f"f_opt (from JSON): {f_opt}\n")

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
        # shifted = values - np.min(values)
        shifted = values - f_opt
        # Compute area under curve (trapezoidal rule, dx=1)
        area = np.trapz(shifted)
        areas.append(area)

    if not areas:
        raise ValueError("No valid lists found in the file.")

    avg_area = np.mean(areas)
    print(f"Average area under the curve across all lists: {avg_area}")
    return avg_area


def parse_filename(file_path):
    """
    Extracts episode, problem, and instance from the npy filename.
    Expected pattern:
        fitness_episode_<E>_problem_<P>_instance_<I>.npy
    """
    filename = os.path.basename(file_path)

    pattern = r"fitness_episode_(\d+)_problem_(\d+)_instance_(\d+)\.npy"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match expected pattern "
            "'fitness_episode_<E>_problem_<P>_instance_<I>.npy'"
        )
    
    episode, problem, instance = map(int, match.groups())
    return episode, problem, instance


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_npy_file>")
        sys.exit(1)
    average_auc(sys.argv[1])
