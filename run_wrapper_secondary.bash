#/bin/bash


if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_base_name>"
    exit 1
fi

EXPERIMENT_BASE="$1"

RESULTS_DIR="output_data/results"

SPACE_DIR="${RESULTS_DIR}/${EXPERIMENT_BASE}_space"

# Get hostname and start time
hostname=$(hostname)
start_time=$(date +"%Y-%m-%d %H:%M:%S")

echo "Script started at: $start_time on machine: $hostname"
echo "Experiment base name: $EXPERIMENT_BASE"



rm -rf "$SPACE_DIR" 
mkdir -p "$SPACE_DIR" 


# Run the Python command for using SPACE, using space number 2which is improvement
python run.py \
    --train \
    --test_models \
    --test_cma_es \
    --test_one_fifth_es \
    --type bbob \
    --instance 1 \
    --dim 40 \
    --experiment_name "${EXPERIMENT_BASE}_space" \
    --use_space 3 \
    --instance_ordering 2 \
    --num_training_instances 12 \
    --num_steps_per_rollout 4800 \
    > "${EXPERIMENT_BASE}_output_space.txt"

# Get end time and duration
end_time=$(date +"%Y-%m-%d %H:%M:%S")
duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))

echo "Script finished at: $end_time on machine: $hostname"
echo "Total runtime: ${duration} seconds"



mv "${EXPERIMENT_BASE}_output_space.txt" "$SPACE_DIR/"

echo "Output files moved to:"
echo "  $SPACE_DIR"
echo "  $DEFAULT_DIR"