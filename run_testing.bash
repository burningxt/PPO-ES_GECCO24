#!/bin/bash


if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_base_name>"
    exit 1
fi

EXPERIMENT_BASE="$1"

RESULTS_DIR="output_data/results"

SPACE_DIR="${RESULTS_DIR}/${EXPERIMENT_BASE}_space"
DEFAULT_DIR="${RESULTS_DIR}/${EXPERIMENT_BASE}_default"

# Get hostname and start time
hostname=$(hostname)
start_time=$(date +"%Y-%m-%d %H:%M:%S")

echo "Script started at: $start_time on machine: $hostname"
echo "Experiment base name: $EXPERIMENT_BASE"


# Run the Python command for using SPACE
python run.py \
    --test_models \
    --type bbob \
    --instance 1 \
    --dim 40 \
    --experiment_name "${EXPERIMENT_BASE}_space" \
    --use_space 1 \
    --num_training_instances 12 \
    > "${EXPERIMENT_BASE}_output_space.txt"



# Run the Python command for not using SPACE
python run.py \
    --test_models \
    --type bbob \
    --instance 1 \
    --dim 40 \
    --experiment_name "${EXPERIMENT_BASE}_default" \
    --use_space 0 \
    --num_training_instances 12 \
    > "${EXPERIMENT_BASE}_output_default.txt"



# Get end time and duration
end_time=$(date +"%Y-%m-%d %H:%M:%S")
duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))

echo "Script finished at: $end_time on machine: $hostname"
echo "Total runtime: ${duration} seconds"



mv "${EXPERIMENT_BASE}_output_space.txt" "$SPACE_DIR/"
mv "${EXPERIMENT_BASE}_output_default.txt" "$DEFAULT_DIR/"

echo "Output files moved to:"
echo "  $SPACE_DIR"
echo "  $DEFAULT_DIR"