#!/bin/bash

# Get hostname and start time
hostname=$(hostname)
start_time=$(date +"%Y-%m-%d %H:%M:%S")

echo "Script started at: $start_time on machine: $hostname"

# Run the Python command
python run.py --instance 1 --dim 40 --type bbob --train --test_models --test_cma_es --test_one_fifth_es

# Get end time and duration
end_time=$(date +"%Y-%m-%d %H:%M:%S")
duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))

echo "Script finished at: $end_time on machine: $hostname"
echo "Total runtime: ${duration} seconds"
