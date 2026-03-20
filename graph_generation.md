
# Single NP file
## Generating graph of single NP file 
To generate a graph of the best solution quality at each generation for a test run. 

    python <graph gen program> <npy file path>
        `python base_graph_single_subarray_including_global_min.py ../output_data/results/final_60_model_default/episodes_tested/DIM_40/fitness_episode_1200_problem_1_instance_1.npy`

## Calculating area under an individual graph
To calculate how much the area under a graph is, using the global minimum as the bottom, to show training progression. 

python data_collection/area_under_graph_to_best_eval/average_under_area_for_one_problem.py output_data/results/final_60_model_space/episodes_tested/DIM_40/fitness_episode_1200_problem_24_instance_1.npy 


changed the name final_60_model to final_baseline_old_restfunc




# From an averaged CSV

## To graph the averagetd values across all 25 runs and all problems for every episode increment. 

### To make the CSV 
python data_collection/average_under_graph_to_best_possible/calc_all_files_normalized.py output_data/results/final_reset_simprovement_space/episodes_tested/DIM_40

### To make the subsequent graph 
python graph_generation_from_already_averaged_file/average_under_one_line_with_policy_updates.py output_data/results/final_60_model_space/space_AUCs.csv 

There is also the other option of the one which also shows policy updates as a red line


# For the Collected Instance Evaluations 

can run:

`python final_graph_generation/graphing_collected_instance_evals.py output_data/results/final_60_model_space/collected_evals_seed_42.csv`