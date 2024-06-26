import cocoex
from src.environment.vanilla_es import ES
from src.config.config import FES_MAX, STATE_SIZE, ACTION_SIZE, POP_SIZE, SIGMA_0, TRAIN_INSTANCE
from src.utilities.plotting import Draw
from src.utilities.tools import find_project_root
import gymnasium as gym
import numpy as np
import os


class ES_Env(gym.Env):
    """
    Custom Environment for CMA-ES that follows gym interface.
    """

    def __init__(self,
                 problem_type="bbob",
                 instance=1,
                 dim=40,
                 fes_max=FES_MAX,
                 sigma_0=SIGMA_0,
                 problem_index=1,
                 seed=None):
        super(ES_Env, self).__init__()
        self.seed(seed)  # Set the seed using the inherited method
        self.suite = cocoex.Suite(problem_type,
                             "",
                             f"dimensions: {dim} function_indices:1-24 instance_indices:{instance}")
        self.problem = self.suite.get_problem(problem_index - 1)
        self.problem_index = problem_index
        self.es = ES(dim, sigma_0, POP_SIZE)
        self.dim = dim
        self.fes_max = fes_max
        self.countevals = 0
        self.current_episode = 0
        self.episode_data = []
        self.fitness_values = None
        self.current_best_fitness = None
        self.previous_best_fitness = None
        self.cumulative_reward = 0
        self.base_dir = find_project_root(os.path.dirname(os.path.abspath(__file__)), 'run.py')

        self.mode = 'training'  # Default mode is training
        self.action_space = gym.spaces.Box(low=-np.ones(ACTION_SIZE), high=np.ones(ACTION_SIZE),
                                           shape=(ACTION_SIZE,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-np.ones(STATE_SIZE), high=np.ones(STATE_SIZE),
                                                shape=(STATE_SIZE,), dtype=np.float32)

    def seed(self, seed=None):
        np.random.seed(seed)

    def set_mode(self, mode):
        self.mode = mode

    def evaluate_fitness(self, solution):
        return self.problem(solution)

    def calculate_improvement_ratio(self):
        if self.previous_best_fitness is None or self.current_best_fitness is None:
            improvement_ratio = 0
        elif self.previous_best_fitness > 0:
            if self.current_best_fitness < 0:
                improvement_ratio = 0.1
            else:
                improvement_ratio = ((self.previous_best_fitness - self.current_best_fitness)
                                     / abs(self.previous_best_fitness))
        elif self.previous_best_fitness < 0:
            improvement_ratio = ((self.previous_best_fitness - self.current_best_fitness)
                                 / abs(self.current_best_fitness))
        elif self.previous_best_fitness == 0:
            improvement_ratio = 0
        return improvement_ratio

    def get_reward(self, improvement_ratio):
        return improvement_ratio

    def norm_input(self):
        log_sigma = np.floor(np.log10(self.es.sigma))
        leading_sigma = self.es.sigma / (10 ** log_sigma)
        norm_sigma = (log_sigma + 0.1 * leading_sigma + 20) / 22.1
        return norm_sigma

    # def norm_input(self):
    #     norm_sigma = (self.es.sigma - 1e-20) / (100 - 1e-20)
    #     return norm_sigma

    def step(self, action):
        solutions = self.es.ask()
        scaling_factor = action[0] * 0.75 + 1.25
        self.es.sigma *= scaling_factor
        self.es.sigma = max(1e-20, min(100.0, self.es.sigma))
        self.fitness_values = np.apply_along_axis(self.problem, 1, solutions)
        self.es.tell(solutions, self.fitness_values)

        current_generation_best_fitness = np.min(self.fitness_values)
        if self.previous_best_fitness is not None:
            self.current_best_fitness = min(self.previous_best_fitness, current_generation_best_fitness)
        else:
            self.current_best_fitness = current_generation_best_fitness


        # Calculate the success of the current step
        if self.previous_best_fitness is not None:
            successful_offspring = sum(1 for f in self.fitness_values if f < self.previous_best_fitness)
            success_ratio = successful_offspring / len(self.fitness_values)
        else:
            success_ratio = 0.2

        improvement_ratio = self.calculate_improvement_ratio()
        norm_sigma = self.norm_input()
        reward = self.get_reward(improvement_ratio)
        self.cumulative_reward += reward
        self.improvement_ratios.append(success_ratio)
        observation = np.array([norm_sigma,
                                success_ratio
                                ])
        self.previous_best_fitness = self.current_best_fitness
        self.countevals += POP_SIZE

        if self.mode == 'training':
            terminated = self.countevals >= self.fes_max
            if terminated:
                self.current_episode += 1
                self.episode_data.append([self.current_episode,
                                          self.current_best_fitness,

                                          self.cumulative_reward])

                # print(f'Episode: {self.current_episode}',
                #       f'Problem: F{self.problem_index}',
                #       f'Best Fitness Value: {self.current_best_fitness}')

                self.problem_index += 1
                if self.problem_index % 12 == 0:
                    self.problem_index = 12
                else:
                    self.problem_index = self.problem_index % 12
                self.problem = self.suite.get_problem(self.problem_index - 1)

                # if self.current_episode % 1200 == 0:
                #     Draw().plot_episode_data(self.base_dir, self.episode_data, self.problem_index)

        elif self.mode == 'testing':
            terminated = self.countevals >= self.fes_max + POP_SIZE * 2
        truncated = False
        return observation, reward, terminated, truncated, {}


    def reset(self, **kwargs):
        self.es = ES(self.dim, SIGMA_0, POP_SIZE)
        self.fitness_values = None
        self.current_best_fitness = None
        self.previous_best_fitness = None
        self.countevals = 0
        self.cumulative_reward = 0
        self.improvement_ratios = []
        return np.zeros(STATE_SIZE)
