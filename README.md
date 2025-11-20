# Package Install List 
codex 
coco-experiment
scipy
matplotlib
gymnasium
stable_baselines3

# To activate venv

start: 
    source venv/bin/activate

# My Custom Experiment

python run.py --instance 1 --dim 40 --type bbob --train --test_models --test_cma_es --test_one_fifth_es

# For training models for my new things
- this will train a model for all the episodes specified in the config on the first 12 problems
python run.py --instance 1 --dim 40 --type bbob --train 