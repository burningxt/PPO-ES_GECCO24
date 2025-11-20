import cocoex as ex
import numpy as np
import json
import os

# Configuration
DIMENSIONS = [40]           # extend this list if you want more dims later
INSTANCES = [1]             # extend this list too if needed
FUNCTIONS = "1-24"
OUTPUT_FILE = "bbob_optima.json"

results = {}

for dim in DIMENSIONS:
    dim_key = f"dim_{dim}"
    results[dim_key] = {}

    for inst in INSTANCES:
        inst_key = f"instance_{inst}"
        results[dim_key][inst_key] = {}

        suite = ex.Suite("bbob", "", f"dimensions:{dim} function_indices:{FUNCTIONS} instance_indices:{inst}")

        for problem in suite:
            # Ask COCO to dump x_opt to file
            problem._best_parameter("print")

            # Load x_opt vector
            x_opt = np.loadtxt("._bbob_problem_best_parameter.txt").tolist()

            # Evaluate true global optimum value
            f_opt = float(problem(x_opt))

            # Identify function number
            func_id, d, instance = problem.id_triple
            func_key = f"f{func_id:02d}"

            # Save to JSON structure
            results[dim_key][inst_key][func_key] = {
                "x_opt": x_opt,
                "f_opt": f_opt
            }

# Cleanup temporary file
if os.path.exists("._bbob_problem_best_parameter.txt"):
    os.remove("._bbob_problem_best_parameter.txt")

# Write to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved results to {OUTPUT_FILE}")
