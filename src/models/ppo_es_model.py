import random

from src.environment.es_env import ES_Env
from src.callbacks.callbacks import LearningRateScheduler, SaveOnBestTrainingRewardCallback, UpdateEnvCallback
from src.config.config import NUM_RUN, TRAIN_INSTANCE, EPISODES
from src.utilities.tools import linear_schedule, save_data
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os


                  #data path  # episodes tested dir
def test_model(env, model_path, data_path, episode, problem_index, instance, experiment_logger):


    experiment_logger.debug(f"About to run tests inside test_model on %d ", problem_index)


    # Loading one of the saved models
    model = PPO.load(model_path, env=env)
    all_fitness_values = []

    # the number of runs, set to 25 in the config
    # the size of episodes is also dictated in the config, 1, 600,
    # it breaks when I change the number of episodes


    for num in range(NUM_RUN):

        experiment_logger.debug(f"Attempting problem run: %d", num)
        temp = env.envs[0].reset()
    

        obs = temp[0]
        # Making it not have an instance in the observation for this prediction 
        obs[2] = 0.0
        first_value = temp[1]
        
        fitness_values = []
        fitness_values.append(first_value)

        while env.envs[0].unwrapped.countevals <= env.envs[0].unwrapped.fes_max:  # Adjust the number of steps as needed

            action, _states = model.predict(obs, deterministic=True)

            obs, rewards, dones, info = env.step(np.array([action]))
            obs = obs[0]
            if dones:
                break
            fitness_values.append(env.envs[0].unwrapped.current_best_fitness)
        all_fitness_values.append(np.array(fitness_values))


    all_fitness_values_arr = np.array(all_fitness_values)

    # Construct the file path
    save_path = os.path.join(
        data_path,
        f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'
    )

    save_data(save_path, all_fitness_values_arr)


class PPO_ES:
    def __init__(self, base_dir, cuda_device, logger=None, config_info=None):
        self.base_dir = base_dir
        self.cuda_device = cuda_device
        self.space_logger = logger
        # self.seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
        
        self.seeds = [42, 789, 1011, 1617, 2021]
        # self.num_models_to_gen = 5; # generating 5 models
        self.num_models_to_gen = 1
        # self.results_dir = os.path.join(base_dir, 'output_data', 'results')
        self.results_dir = os.path.join(base_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        print(config_info)
        self.config_info = config_info

    # this contains the entire code for the PPO+ES training loop

    def train_ppo_es(self):
        self.space_logger.info(f"Starting training")
        # we try for every seed
        # we train 10 independent PPO agents (one per seed)
        # for seed in self.seeds:
        # for _ in range(self.num_models_to_gen): # just generates 5 differnet random models with different seeds for now
        for seed in self.seeds:
            # using a different random seed on each run instead

            """ 
            need to plan how im going to call each training step with each instance here 
            """
            # added to make the seeds random values
            # seed = random.randint(1,9999)
            self.space_logger.info("------------------------------")
            self.space_logger.info(f"STARTING TO TRAIN A NEW MODEL")
            self.space_logger.info(f"   training with seed: %d", seed)
            self.space_logger.info("------------------------------")
            # create the environment: 
                # make_vec_env: war4ps the ES_Env into a vectorized envionment, so it can be used by PPO
                # ES_Env: Is the custom envrionment for the Evolution Strategies Problem 
                # set the seed
                                                    # right now we only train on instance 1 as outlined by config
                    
            print(self.config_info)
            # THIS CALLS THE CONSTRUCTOR WHICH STARTS THE TRAINING, THIS IS WHERE I'LL NEED FOR SPACE
            env = make_vec_env(lambda: ES_Env( 
                                              seed=seed, 
                                              space_logger=self.space_logger,
                                              dim=self.config_info["test_dimension"], 
                                              use_space=self.config_info["use_space"], 
                                              num_training_instances=self.config_info["num_training_instances"], 
                                              instance=self.config_info["test_instance"]), # Instance used to just always be instance 1, no more
                               
                                            n_envs=1)
                                                # train instance is always 1 for their experiments

            N_STEPS = 12 * 400
            # Reset the model with the new environment to ensure it's training from scratch
            model = PPO(
                policy='MlpPolicy',
                env=env, # passing in our environment 
                device=self.cuda_device,
                learning_rate=3e-4,
                verbose=1,
                # n_steps=12 * 400,  # Number of steps to run for each environment per update
                n_steps=N_STEPS, 
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
            callback = SaveOnBestTrainingRewardCallback(save_path=episodes_trained_dir, seed=seed, space_logger=self.space_logger)
            total_timesteps = 12 * 4000

            # adjusts the learning rate linearly over time, decays from 3e-4 to 0 by the end of training 
            scheduler = linear_schedule(initial_value=3e-4)
            lr_scheduler_callback = LearningRateScheduler(
                initial_learning_rate=3e-4,
                scheduler=scheduler
            )

            

            lr_scheduler_callback.total_timesteps = total_timesteps
            # only takes PPO for now
            space_callback = UpdateEnvCallback("ppo", space_logger=self.space_logger, use_space=self.config_info["use_space"])
            # model.learn 
            # model is a PPO object, so it just learns 
            # we import PPO from stable baseline 3
                # the callback returns the trained model after neraling
                # https://stable-baselines3.readthedocs.io/en/v1.0/modules/ppo.html
                # https://spinningup.openai.com/en/latest/algorithms/ppo.html
            # train the model 
            # callback, lets you inject custom logic            # this callback used to save the best one             # lr_scheduler_callback is used to decrease the learning rate
            model.learn(total_timesteps=total_timesteps, callback=[callback, lr_scheduler_callback, space_callback]) # it calls the callback when training ends 
            """
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
            """

            # can see where step is defined inside the venv/lib/python3.9/site-packages/gymnasium

            

    # This uses the numpy data ive been looking at 
    def test_ppo_es(self, problem_type, test_problem_dimension, problem_index, instance, experiment_logger):
        # Randomly select one seed for testing
        # I cant randomly select one seed if my seeds are random
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
                                               seed=seed, debug_logger=experiment_logger

                                            #   space_logger=self.space_logger,
                                               ), n_envs=1)


        experiment_logger.debug(f"About to run all episodes for %d problem inside test_ppo_es", problem_index)
        # generate for each of hte episodes
        # here is where the number of episodes is decided 
        for episode in EPISODES:
            model_filename = f"model_seed_{seed}_episode_{episode}.zip"

            # we store our model here 
            test_model_path = os.path.join(self.results_dir, 'episodes_trained', model_filename)

            for single_env in seed_env.envs:
                single_env.unwrapped.set_mode('testing')



            test_model(seed_env, test_model_path, episodes_tested_dir, episode, problem_index, instance, experiment_logger)

