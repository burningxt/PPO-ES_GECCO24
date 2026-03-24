# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2024

### Added - Vincent Pickering's Dissertation Work

- **SPACE Curriculum Learning System** (`src/callbacks/callbacks.py`)
  - Self-paced curriculum learning for ES policy optimization
  - Dynamic curriculum size adjustment based on learning stability
  - Multiple instance ordering strategies (absolute, improvement, relative improvement)

- **Enhanced State Space** (`src/environment/es_env.py`)
  - Added one-hot encoding for 12 training instances
  - Extended STATE_SIZE from 2 to 14 dimensions
  - Support for curriculum management methods

- **Flexible Training Configuration** (`src/config/config.py`)
  - Extended EPISODES list for finer-grained control (60-episode intervals)
  - Added space_enum for SPACE operation modes

- **Data Collection and Analysis Tools**
  - `data_collection/` - Scripts for computing area under curve and performance metrics
  - `graph_generation/` - Scripts for generating publication-quality plots
  - Enhanced logging system with separate log files for training, models, and debug info

- **New Command Line Arguments** (`run.py`)
  - `--experiment_name` - Name experiments for organized output
  - `--use_space` - Enable/disable SPACE curriculum
  - `--instance_ordering` - Choose instance ordering strategy
  - `--num_training_instances` - Control training set size
  - `--num_steps_per_rollout` - Configure rollout length

### Changed

- Updated model saving frequency to every 60 episodes (was every 120)
- Improved logging with structured format (timestamp, level, module, message)
- Refactored ES environment to support curriculum-based training

### Removed

- Cleaned up repository by removing ~1400 old baseline `.npy` data files
- Removed unused output files and temporary data

### Fixed

- Added proper error handling for model saving
- Fixed state representation to include instance information

## [Original] - GECCO 2024

- Initial PPO-ES implementation
- BBOB benchmark support
- CMA-ES and 1/5-ES baselines
