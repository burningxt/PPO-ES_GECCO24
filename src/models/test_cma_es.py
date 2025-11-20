import cma
import cocoex
from src.config.config import FES_MAX, NUM_RUN, POP_SIZE, SIGMA_0
from src.utilities.tools import save_data
import numpy as np
import os


def test_cma_es(results_dir, problem_type, test_problem_dimension, problemIndex, instance):
    suite = cocoex.Suite(problem_type,
                         "",
                         f"dimensions: {test_problem_dimension} function_indices:1-24 instance_indices:{instance}")
    problem = suite.get_problem(problemIndex - 1)
    # Create a new directory for each problem index

    baselines_dir = os.path.join(results_dir, f'baselines', f'DIM_{test_problem_dimension}')
    os.makedirs(baselines_dir, exist_ok=True)

    all_fitness_values = []

    for _ in range(NUM_RUN):
        # TODO may want to add a similar thing here for printing the initial solution/vector before any evaluation
        cma_es = cma.CMAEvolutionStrategy(test_problem_dimension * [0], SIGMA_0, {'popsize': POP_SIZE})
        lst_fitness_values = []
        fes = 0
        current_best_fitness = None
        while fes <= FES_MAX:
            solutions = cma_es.ask()
            fitness_values = np.apply_along_axis(problem, 1, solutions)
            cma_es.tell(solutions, fitness_values)
            current_generation_best_fitness = np.min(fitness_values)
            if current_best_fitness is not None:
                current_best_fitness = min(current_best_fitness, current_generation_best_fitness)
            else:
                current_best_fitness = current_generation_best_fitness
            fes += POP_SIZE
            lst_fitness_values.append(np.array(current_best_fitness))
        all_fitness_values.append(lst_fitness_values)
    all_fitness_values_arr = np.array(all_fitness_values)
    save_data(os.path.join(baselines_dir, f'fitness_cma_es_problem_{problemIndex}_instance_{instance}.npy'), all_fitness_values_arr)


