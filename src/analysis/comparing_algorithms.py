from src.models.ppo_es_model import PPO_ES
from src.models.test_cma_es import test_cma_es
from src.models.test_one_fifth_es import test_one_fifth_es
from src.utilities.plotting import Draw
from src.utilities.friedman import perform_friedman_test
from src.utilities.tools import find_project_root
import os
import json
from datetime import datetime
import time
import logging

import platform
import socket

def comparing_algorithms(need_train=False,
                         test_problem_type='bbob',
                         test_instance=1,
                         test_dimension=40,
                         need_test_models=False,
                         need_test_cma_es=False,
                         need_test_one_fifth_es=False,
                         cuda_device='cuda:0',
                         experiment_name='base',
                         use_space=1,
                         instance_ordering=1,
                        #  use_default=0,
                         num_training_instances=12,
                         num_steps_per_rollout=12*400 # default is 12 * 400 which was the original value

                         ):
    
    """Print all comparison parameters to standard output."""
    print("=== Algorithm Comparison Parameters ===")
    print(f"Train: {need_train}")
    print(f"Problem Type: {test_problem_type}")
    print(f"Instance: {test_instance}")
    print(f"Dimension: {test_dimension}")
    print(f"Test Models: {need_test_models}")
    print(f"Test CMA-ES: {need_test_cma_es}")
    print(f"Test One-Fifth ES: {need_test_one_fifth_es}")
    print(f"CUDA Device: {cuda_device}")
    print("=======================================")



    base_dir = find_project_root(os.path.dirname(os.path.abspath(__file__)), 'run.py')
    results_dir = os.path.join(base_dir, 'output_data', 'results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    plot_dir = os.path.join(results_dir, 'plots', f'DIM_{test_dimension}', f'instance_{test_instance}')
    os.makedirs(plot_dir, exist_ok=True)


    machine_info = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor()
    }



    config = {
        "system_info": machine_info,
        "starting timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "need_train": need_train,
        "test_problem_type": test_problem_type,
        "test_instance": test_instance,
        "test_dimension": test_dimension,
        "need_test_models": need_test_models,
        "need_test_cma_es": need_test_cma_es,
        "need_test_one_fifth_es": need_test_one_fifth_es,
        "cuda_device": str(cuda_device),
        "experiment_name": experiment_name,
        "use_space": use_space,
        "instance_ordering": instance_ordering,
        "num_training_instances": num_training_instances,
        "num_steps_per_policy_update":num_steps_per_rollout,
  
    }

    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Saved experiment configuration to: {config_path}")

    # Create a logger
    experiment_logger = build_logger(results_dir)

    # changed from being base_dir passsed, will this be a problem? 
    ppo_es = PPO_ES(base_dir=results_dir, cuda_device=cuda_device, logger=experiment_logger, config_info=config) # cuda is for parallel computing 
    start_time = time.time()

    try: 
        # training called here and here alone
        # its only trained on the first 12 I believe 
        if need_train:
            ppo_es.train_ppo_es(num_steps_per_rollout)

        # Create new directories for each problem index
        episodes_tested_dir = os.path.join(results_dir, f'episodes_tested', f'DIM_{test_dimension}')
        baselines_dir = os.path.join(results_dir, f'baselines', f'DIM_{test_dimension}')


        print("creating episodes_tested_dir")
        print(episodes_tested_dir)
        os.makedirs(episodes_tested_dir, exist_ok=True)
        os.makedirs(baselines_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)




        for problemIndex in range(1, 25): # iterates through 1 to 24 
            if need_test_models:
                
                experiment_logger.debug(f"Calling test_ppo_es on problem_index: %d", problemIndex)
                ppo_es.test_ppo_es(test_problem_type, test_dimension, problemIndex, test_instance, experiment_logger)
            if need_test_cma_es:
                test_cma_es(results_dir, test_problem_type, test_dimension, problemIndex, test_instance)
            if need_test_one_fifth_es:
                test_one_fifth_es(results_dir, test_problem_type, test_dimension, problemIndex, test_instance)
            # plot convergence figure
            Draw().plot_convergence_data_mean_ci(problemIndex, episodes_tested_dir, baselines_dir, plot_dir, test_instance)

        # plot overall box figure
        Draw().plot_standardized_performance_boxplot(episodes_tested_dir, baselines_dir, plot_dir, test_instance)

        # generate Friedman test table
        perform_friedman_test(episodes_tested_dir, baselines_dir, plot_dir, test_instance)
    
    finally: 

        end_time = time.time()
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_runtime_seconds = round(end_time - start_time, 2)

        # Reload config, update it
        with open(config_path, "r") as f:
            config = json.load(f)

        config["end_timestamp"] = end_timestamp
        config["total_runtime_seconds"] = total_runtime_seconds

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)




def build_logger(experiment_dir: str):
    os.makedirs(experiment_dir, exist_ok=True)

    training_log_path = os.path.join(experiment_dir, "training.log")
    debug_log_path = os.path.join(experiment_dir, "debug.log")
    models_log_path = os.path.join(experiment_dir, "models.log")

    # Main logger
    logger = logging.getLogger("ppo_es_training")
    logger.setLevel(logging.DEBUG)  # allow debug through the logger itself

    # If you run this multiple times in the same process, avoid duplicate handlers
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # --- training.log: INFO+ ---
    training_handler = logging.FileHandler(training_log_path)
    training_handler.setLevel(logging.INFO)
    training_handler.setFormatter(formatter)
    training_handler.set_name("training_file")
    logger.addHandler(training_handler)

    # --- debug.log: DEBUG only (no INFO duplication) ---
    debug_handler = logging.FileHandler(debug_log_path)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    debug_handler.set_name("debug_file")
    debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
    logger.addHandler(debug_handler)

    # --- child logger for models ---
    models_logger = logger.getChild("modellog")
    models_logger.setLevel(logging.INFO)

    # Important: models_logger is its own logger object, clear its handlers too
    models_logger.handlers.clear()

    models_handler = logging.FileHandler(models_log_path)
    models_handler.setLevel(logging.INFO)
    models_handler.setFormatter(formatter)
    models_handler.set_name("models_file")
    models_logger.addHandler(models_handler)

    # Prevent modellog from also writing into training.log/debug.log via propagation
    models_logger.propagate = False

    # Attach for convenience
    logger.modellog = models_logger

    # Prevent main logger from propagating to root logger (avoids console duplicates)
    logger.propagate = False

    return logger
