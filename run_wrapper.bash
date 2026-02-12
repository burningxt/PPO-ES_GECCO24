#!/bin/bash

# Get hostname and start time
hostname=$(hostname)
start_time=$(date +"%Y-%m-%d %H:%M:%S")

echo "Script started at: $start_time on machine: $hostname"

# Run the Python command for using SPACE
python run.py \
    --train \
    --test_models \
    --test_cma_es \
    --test_one_fifth_es \
    --type bbob \
    --instance 1 \
    --dim 40 \
    --experiment_name full_bbob_dim40_space \
    --use_space 1 \
    --num_training_instances 12

# Run the Python command for not using SPACE
python run.py \
    --train \
    --test_models \
    --test_cma_es \
    --test_one_fifth_es \
    --type bbob \
    --instance 1 \
    --dim 40 \
    --experiment_name full_bbob_dim40_default \
    --use_space 0 \
    --num_training_instances 12



# Get end time and duration
end_time=$(date +"%Y-%m-%d %H:%M:%S")
duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))

echo "Script finished at: $end_time on machine: $hostname"
echo "Total runtime: ${duration} seconds"
