# Package Install List 
codex 
coco-experiment
scipy
matplotlib
gymnasium
stable_baselines3

# To activate venv

create:
    python -m venv venv

start: 
    source venv/bin/activate

# My Custom Experiment

python run.py --instance 1 --dim 20 --type bbob --train --test_models --test_cma_es --test_one_fifth_es