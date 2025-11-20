import random

from src.environment.es_env import ES_Env
from src.callbacks.callbacks import LearningRateScheduler, SaveOnBestTrainingRewardCallback
from src.config.config import NUM_RUN, TRAIN_INSTANCE, EPISODES
from src.utilities.tools import linear_schedule, save_data
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os


# FLAGVP: This is where the data is saved for the numply files ive been looking at 
# this is what I need to gain a deeper understanding of 
# this is called from the test_ppo_es below 
def test_model(env, model_path, data_path, episode, problem_index, instance):

    # ppo is an import
    # we learn the model_path
    # model path here is going to be one of the ones in episodes trained
    model = PPO.load(model_path, env=env)
    all_fitness_values = []

    # the number of runs, set to 25 in the config
    # the size of episodes is also dictated in the config, 1, 600,
    # it breaks when I change the number of episodes
    starting_values = []
    for _ in range(NUM_RUN):
        # the only call to .reset()
        temp = env.envs[0].reset()
        obs = temp[0]
        first_value = temp[1]
        
        fitness_values = []
        # adds the starting value to all the output data, so that way data will be more accurate
        fitness_values.append(first_value)
        # starting_values.append(first_value)

        # we use hte mode created, which is in es_env.py 
            # env is environment/es_env 
        while env.envs[0].unwrapped.countevals <= env.envs[0].unwrapped.fes_max:  # Adjust the number of steps as needed
            # this model.predict, is where the model predicts what hte step size should be 
            action, _states = model.predict(obs, deterministic=True)

            # we do the step function, which moves it forward
                # sets the current_best_fitness to the min of prev and current 
                # current generation best fitness is just the minimum of the current fitness values
            obs, rewards, dones, info = env.step(np.array([action]))
            obs = obs[0]
            if dones:
                break
            fitness_values.append(env.envs[0].unwrapped.current_best_fitness)
        all_fitness_values.append(np.array(fitness_values))


    # this is the array that we have been looking at 
    all_fitness_values_arr = np.array(all_fitness_values)
    # This is where it gets those names from 

    # where it writes to the numpy array 
    save_data(os.path.join(data_path, f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'), all_fitness_values_arr)


class PPO_ES:
    def __init__(self, base_dir, cuda_device):
        self.base_dir = base_dir
        self.cuda_device = cuda_device
        self.seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
        self.num_models_to_gen = 5; # generating 5 models
        # Initialize paths for saving results and plots
        # we have a results dictionary 
        self.results_dir = os.path.join(base_dir, 'output_data', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    # this contains the entire code for the PPO+ES training loop

    def train_ppo_es(self):
        # we try for every seed
        # we train 10 independent PPO agents (one per seed)
        # for seed in self.seeds:
        for _ in range(self.num_models_to_gen):
            # using a different random seed on each run instead
            seed = random.randint(1,9999)
            # create the environment: 
                # make_vec_env: war4ps the ES_Env into a vectorized envionment, so it can be used by PPO
                # ES_Env: Is the custom envrionment for the Evolution Strategies Problem 
                # set the seed
            env = make_vec_env(lambda: ES_Env(instance=TRAIN_INSTANCE, seed=seed), n_envs=1)
            # Reset the model with the new environment to ensure it's training from scratch
            model = PPO(
                policy='MlpPolicy',
                env=env, # passing in our environment 
                device=self.cuda_device,
                learning_rate=3e-4,
                verbose=1,
                n_steps=12 * 400,  # Number of steps to run for each environment per update
                batch_size=64,  # Batch size for training
                n_epochs=10,  # Number of epochs to run for each update
                gamma=0.99,  # Discount factor for the reward function
                gae_lambda=0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                                  # Smooths advantage estimates
                clip_range=0.2 # PPO clipping threshold for stability
            )
            episodes_trained_dir = os.path.join(self.results_dir, f'episodes_trained')
                # make the output directory for the model directory if its not already there 
            os.makedirs(episodes_trained_dir, exist_ok=True)

            # montiros the mean reward during training. if a new "best" reward is reached, saves the current model weights. 
                # so you load the best checkpoint instead of the last one
                # these callbacks are defined in the callbacks class of this implementation 
            # this callback is the thing that saves the model to the .zip file
            callback = SaveOnBestTrainingRewardCallback(save_path=episodes_trained_dir, seed=seed)
            total_timesteps = 12 * 4000

            # adjusts the learning rate linearly over time, decays from 3e-4 to 0 by the end of training 
            scheduler = linear_schedule(initial_value=3e-4)
            lr_scheduler_callback = LearningRateScheduler(
                initial_learning_rate=3e-4,
                scheduler=scheduler
            )

            lr_scheduler_callback.total_timesteps = total_timesteps
            # model.learn 
            # model is a PPO object, so it just learns 
            # we import PPO from stable baseline 3
                # the callback returns the trained model after neraling
                # https://stable-baselines3.readthedocs.io/en/v1.0/modules/ppo.html
                # https://spinningup.openai.com/en/latest/algorithms/ppo.html
            # train the model 
            # callback, lets you inject custom logic            # this callback used to save the best one             # lr_scheduler_callback is used to decrease the learning rate
            model.learn(total_timesteps=total_timesteps, callback=[callback, lr_scheduler_callback]) # it calls the callback when training ends 
            # this is just a method from stablebaselines3 
                # runs the environment for n_steps timesteps
                # computes advantage estimates
                # step() is manually called inside the while loop in this file 
                    # it is called no where else 
            # runs in the environment we made earlier
            # RL environments must follow the Gym API 
            # step: just takes in action 
                # applies action 
                # returns the new state
                # returns reward
                # reutrns done if done
                # info contains extra info 
                # return observation, reward, terminated, truncated, {}

            # can see where step is defined inside the venv/lib/python3.9/site-packages/gymnasium

            

    # This uses the numpy data ive been looking at 
    def test_ppo_es(self, problem_type, test_problem_dimension, problem_index, instance):
        # Randomly select one seed for testing
        seed = random.choice(self.seeds)
        # results dir is just made as part of the PPO-ES object 
        episodes_tested_dir = os.path.join(self.results_dir, f'episodes_tested', f'DIM_{test_problem_dimension}')
        # here is where we create the directory
        os.makedirs(episodes_tested_dir, exist_ok=True)

        # seed env 
            # make_vec_env is a utility of the stable baselines library
                # creates a vectorized, environment for training reinforcement learning agents 
                # we create one of type ES_env 
                # we test ppo_es in the ES_env environment 
        seed_env = make_vec_env(lambda: ES_Env(problem_type=problem_type, 
                                               instance=instance,
                                               dim=test_problem_dimension,
                                               problem_index=problem_index,
                                               seed=seed), n_envs=1)

        # generate for each of hte episodes
        # here is where the number of episodes is decided 
        for episode in EPISODES:
            model_filename = f"model_seed_{seed}_episode_{episode}.zip"

            # we store our model here 
            test_model_path = os.path.join(self.results_dir, 'episodes_trained', model_filename)

            for single_env in seed_env.envs:
                single_env.unwrapped.set_mode('testing')

            # we have this test model 
            # I think the env is in es_env 
            test_model(seed_env, test_model_path, episodes_tested_dir, episode, problem_index, instance)

