import numpy as np
import sys
import os
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# JSON_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "bbob_optima.json"))
JSON_PATH = "/cs/home/vap2/Documents/diss/PPO-ES_GECCO24/bbob_optima.json"



def load_optima():
    with open(JSON_PATH, "r") as f:
        return json.load(f)


def get_f_opt(optima, dim, instance, problem_index):
    dim_key = f"dim_{dim}"
    inst_key = f"instance_{instance}"
    func_key = f"f{problem_index:02d}"
    return float(optima[dim_key][inst_key][func_key]["f_opt"])


def parse_filename(file_path):
    filename = os.path.basename(file_path)
    pattern = r"fitness_episode_(\d+)_problem_(\d+)_instance_(\d+)\.npy"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    episode, problem, instance = map(int, match.groups())
    return episode, problem, instance


def get_dimension_from_path(file_path):
    parent = os.path.basename(os.path.dirname(file_path))
    match = re.match(r"DIM_(\d+)", parent)
    if not match:
        raise ValueError(f"Parent directory does not match expected DIM_<dim> pattern: {parent}")
    return int(match.group(1))


def average_auc(file_path, clamp_below_fopt=True):
    data = np.load(file_path, allow_pickle=True)
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("Expected a list or array of lists in the .npy file.")

    # Parse once per file
    dim = get_dimension_from_path(file_path)
    _, problem, instance = parse_filename(file_path)

    optima = load_optima()
    f_opt = get_f_opt(optima, dim, instance, problem)

    areas = []

    for i, lst in enumerate(data):
        if not isinstance(lst, (list, np.ndarray)) or len(lst) == 0:
            print(f"Skipping list {i}: invalid or empty.")
            continue

        values = np.asarray(lst, dtype=float)

        # Shift curve so f_opt is at 0
        shifted = values - f_opt

       
        area = np.trapz(shifted)
        areas.append(area)

    if not areas:
        raise ValueError("No valid lists found in the file.")

    avg_area = float(np.mean(areas))
    print(f"Using f_opt={f_opt} (dim={dim}, instance={instance}, f={problem:02d})")
    print(f"Average area under the curve across all lists: {avg_area}")
    return avg_area


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_npy_file>")
        sys.exit(1)

    average_auc(sys.argv[1])
