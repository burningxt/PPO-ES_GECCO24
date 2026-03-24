# PPO-ES with SPACE Curriculum Learning

This repository contains the implementation of **PPO-ES** (Proximal Policy Optimization for Evolution Strategies) with **SPACE** (Self-Paced Curriculum Learning) for black-box optimization on the BBOB benchmark.

> **Note**: This is an enhanced version with Vincent Pickering's dissertation work on curriculum learning for ES policy optimization.

## Dependencies

```bash
pip install coco-experiment scipy matplotlib gymnasium stable-baselines3 cma
```

Or install individually:
- `coco-experiment` - COCO benchmarking framework
- `scipy` - Scientific computing
- `matplotlib` - Plotting library
- `gymnasium` - RL environment interface
- `stable-baselines3` - PPO implementation
- `cma` - CMA-ES baseline

## Quick Start

### Training with SPACE (Default)

Train a model with SPACE curriculum learning on the first 12 BBOB instances:

```bash
python run.py --instance 1 --dim 40 --type bbob --train \
    --experiment_name "my_experiment" \
    --use_space 1 \
    --instance_ordering 1
```

### Training without SPACE (Original Behavior)

```bash
python run.py --instance 1 --dim 40 --type bbob --train \
    --experiment_name "baseline" \
    --use_space 0
```

### Testing Trained Models

```bash
python run.py --instance 1 --dim 40 --type bbob --test_models \
    --test_cma_es --test_one_fifth_es
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train` | False | Train new models |
| `--test_models` | False | Test trained PPO-ES models |
| `--test_cma_es` | False | Include pure CMA-ES baseline |
| `--test_one_fifth_es` | False | Include 1/5-ES baseline |
| `--experiment_name` | "" | Name for the experiment (used in output paths) |
| `--use_space` | 1 | Enable SPACE curriculum (0=disabled, 1=enabled) |
| `--instance_ordering` | 1 | Instance ordering strategy (0=absolute, 1=improvement, 2=relative_improvement, 3=none) |
| `--num_training_instances` | 12 | Number of BBOB instances to train on |
| `--num_steps_per_rollout` | 4800 | Steps per policy update (rollout length) |
| `--dim` | 40 | Problem dimension |
| `--type` | "bbob" | Benchmark type |

## SPACE Curriculum Learning

SPACE (Self-Paced Curriculum Learning) dynamically adjusts the training difficulty:

1. **Curriculum Size**: Starts with 1 instance, grows/shrinks based on learning stability
2. **Instance Ordering**: Orders instances by difficulty (q-values or improvement)
3. **Stability Detection**: Monitors policy updates to determine curriculum expansion

### Instance Ordering Strategies

- **Absolute (0)**: Orders by absolute q-values
- **Improvement (1)**: Orders by improvement since last evaluation
- **Relative Improvement (2)**: Orders by relative improvement
- **None (3)**: Random ordering

### Space Operation Modes

- **NO_SPACE (0)**: Original behavior without curriculum
- **JUST_SIZES (1)**: Curriculum size adjustment only
- **INSTANCE_STATE (2)**: Full SPACE with instance state
- **ONE_GENERATION (3)**: Single generation mode

## Output Structure

```
output_data/results/{experiment_name}/
├── episodes_trained/          # Saved models
│   ├── model_seed_{seed}_episode_{ep}.zip
├── training.log               # Training progress
├── models.log                 # Model saving events
├── debug.log                  # Debug information
└── csvs/                      # Performance data
```

## Data Analysis

### Area Under Curve Analysis

```bash
python data_collection/area_under_graph_to_best_eval/calc_all_files.py
```

### Generate Comparison Graphs

```bash
python graph_generation/final_value_graphs/all_problem_line_on_same_graph.py
```

## Repository Structure

```
├── src/
│   ├── analysis/              # Algorithm comparison scripts
│   ├── callbacks/             # Training callbacks (SPACE implementation)
│   ├── config/                # Configuration and enums
│   ├── environment/           # ES environment (Gymnasium)
│   ├── models/                # PPO-ES model definitions
│   └── utilities/             # Helper functions
├── data_collection/           # Data analysis scripts
├── graph_generation/          # Plotting scripts
└── run.py                     # Main entry point
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{your_paper_2024,
  title={PPO-ES with SPACE Curriculum Learning},
  author={...},
  booktitle={...},
  year={2024}
}
```

## License

See LICENSE file for details.

## Acknowledgments

Thanks to Vincent Pickering for contributing the SPACE curriculum learning implementation.
