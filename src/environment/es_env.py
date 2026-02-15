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
                 seed=None, 
                 space_logger=None,

                 use_space=1, # use space by default
                 num_training_instances=12, # use first 12 instances for training by default
                 ):
        super(ES_Env, self).__init__() # inheriting the constructor 
        self.seed(seed)  # Set the seed using the inherited method
                                  # the problem type is just bbob by default 

# instance=TRAIN_INSTANCE, seed=seed, space_logger=self.space_logger,dim=self.config_info.test_dimension, space=self.config_info.use_space, num_training_instances=self.config.num_training_instances), n_envs=1)
        # Create a new cocoex suite
        self.suite = cocoex.Suite(problem_type,
                             "",# empty string for the benchmark options (we don't use any)  # instance is 1 by default
                             f"dimensions: {dim} function_indices:1-24 instance_indices:{instance}")
        # Select which problem from the suite, by index (0 based indexing is used internally)

        # Initialize and set basic default value to curriculum
        self.curriculum = [problem_index]

        self.curriculum_size = 1
        self.curriculum_index = 0
        self.space_logger = space_logger


        # Initial Things Added
        self.use_space = use_space
        self.num_training_instances = num_training_instances

        # Setting the initial problem 
        self.problem = self.suite.get_problem(problem_index - 1)
        self.problem_index = problem_index

        # we make the ES here 
        # so the first evaluated population is POP_SIZE 25 samples drawn from a normal distribution sampled at sigma_0 which is .5
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

    # def set_curriculum_index(self, curriculum_index: int):
    #     self.curriculum_index = curriculum_index 

# Should never be setting problem index
    def set_curriculum_index(self, curriculum_index: int):
        self.curriculum_index = curriculum_index
        self.problem = self.suite.get_problem(self.curriculum[curriculum_index-1]) # for the correct indesxing
        self.space_logger.info("Just set problem to: %d in (set_curriculum_index)", self.curriculum[curriculum_index-1])
        self.space_logger.info("Which corresponds to: %s", self.problem)


    def reset_curriculum_index(self):
        self.curriculum_index = 1 

    def set_curriculum(self, problem_list):
        self.curriculum = problem_list
        # Always sets the curriculum index to be 
        self.current_index = 0
        self.problem = self.suite.get_problem(self.curriculum[0])

    def set_curriculum_size(self, size: int):
        self.curriculum_size = size

    def next_problem_from_curriculum(self):
        self.curriculum_index = (self.curriculum_index + 1) % len(self.curriculum)
        # Sets the problem index to whatever is held in the curriculum array 
        self.set_curriculum_index(self.curriculum[self.curriculum_index])

    def get_curriculum(self):
        # Returns a copy of the curriculum list
        return self.curriculum.copy()


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


    # Called from the Gym environment itself
    def step(self, action):

        # self.space_logger.info(f"Curriculm is currently %s", self.curriculum)

        solutions = self.es.ask()
        scaling_factor = action[0] * 0.75 + 1.25
        self.es.sigma *= scaling_factor
        self.es.sigma = max(1e-20, min(100.0, self.es.sigma))

        # Calculate fitness values for solutions returned by ask
        self.fitness_values = np.apply_along_axis(self.problem, 1, solutions)
            # Each row is a candidate solution vector (point in the search space) 
                # Evaluates the candidate solutions on the actual problem
            # 1 means that we apply it to the 1 axis in each vector
            # solutions is a 2D NumPy array
        self.es.tell(solutions, self.fitness_values) # 
        # gonna have the np.min(self.fitness_values), the best of this generation 
        current_generation_best_fitness = np.min(self.fitness_values)

        # only do this comparrison if its not the first generation 
        if self.previous_best_fitness is not None:
            # compare the best fitness of the previous best fitness, and the current best fitness 
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
        self.countevals += POP_SIZE # Track total number of evaluations

        # Added for potentially needed additional information
        infos = {}

        if self.mode == 'training':

                                        
            terminated = self.countevals >= self.fes_max # Why we have 41 (best evals) in each data file

            if terminated:
                # current switching system is accurate, but this number is one below what it really is

                ############### Logging ###################
                self.space_logger.info(f"-----------")
                # self.space_logger.info(f"#")
                # self.space_logger.info(f"Curriculm is currently %s", self.curriculum)
                # self.space_logger.info(f"Curriclum index is currently %s", self.curriculum_index)
                # self.space_logger.info(f"Curriclum at that index is currently %s", self.curriculum[self.curriculum_index-1])
                # self.space_logger.info(f"#")
                self.space_logger.info(f"Problem just trained on was %s", self.problem)
                # self.space_logger.info(f"Episode [%d] completed, switching to function [%d].", self.current_episode,(self.curriculum[self.curriculum_index-1]))
                # self.space_logger.info(f"That being: %s", self.problem)


                self.current_episode += 1
                self.episode_data.append([self.current_episode,
                                          self.current_best_fitness,
                                          self.cumulative_reward])

                # Maintain backwards compatibility for default behavior (no space)
                if not self.use_space:
                    self.problem_index += 1

                    if self.problem_index % self.num_training_instances == 0:
                        self.problem_index = self.num_training_instances 

                    else:
                        self.problem_index = self.problem_index % self.num_training_instances 

                    
                    self.problem = self.suite.get_problem(self.problem_index -1)

                else:

                    # Gets the next problem from the current curriculum
                        # curriculum index is 1 indexed even though curriculum is 0 indexed
                        # the BBOB functions themselves are 0 indexed, so shouldn't need a -1 here 
                    # 
                    if self.curriculum_index >= len(self.curriculum):
                        self.space_logger.info("Curriclum now empty in ES.env, going to be changed soon ")
                        self.space_logger.info(f"--------------")
                    else:

                        self.problem = self.suite.get_problem(self.curriculum[self.curriculum_index]) # don't need a -1 at the end because BBOB is already 0 indexed
                        # self.space_logger.info(f"Testing: %s", self.curriculum)
                        self.space_logger.info("now about to train on: ")
                        self.space_logger.info(f"Curr Problem: %s", self.problem)

                        self.space_logger.info(f"--------------")



                    self.curriculum_index += 1



        elif self.mode == 'testing':
            # here is where it is for testing
            terminated = self.countevals >= self.fes_max + POP_SIZE * 2
        truncated = False

        return observation, reward, terminated, truncated, infos


# the env in space returns obs[observation]

    # reset for PPO-ES
    # have reset back to original 
    # def reset(self, **kwargs):
    #     self.es = ES(self.dim, SIGMA_0, POP_SIZE)
    #     self.fitness_values = None
    #     self.current_best_fitness = None
    #     self.previous_best_fitness = None
    #     self.countevals = 0
    #     self.cumulative_reward = 0
    #     self.improvement_ratios = []

    #     ## Data Handling, evaluating the very first solution manually
    #     first_solution = self.es.xmean.copy()       # this is the "center" at generation 0
    #     # the xmean is the object used for the solutions
    #     first_value = self.problem(first_solution)
    #     return np.zeros(STATE_SIZE), first_value

# trying new reset function 
    def reset(self, **kwargs):
        self.es = ES(self.dim, SIGMA_0, POP_SIZE)
        self.countevals = 0
        self.cumulative_reward = 0
        self.improvement_ratios = []
        self.previous_best_fitness = None
        self.current_best_fitness = None

        prev_problem = self.problem

        # Sets it to be the current first problem in the curriculum
        self.problem = self.suite.get_problem(self.curriculum[0])

        # Same as prev reset()
        first_solution = self.es.xmean.copy() # iss the 'center' of the ES distribution at generation 0
        first_value = float(self.problem(first_solution))

        # Creates an observation vector of fixed size
        obs = np.zeros(STATE_SIZE, dtype=np.float32)
        # The current normalized step size
        obs[0] = self.norm_input()                   #sigma 
        # obs[0] = first_value

        # Problem Difficulty Signal
        # obs[1] = np.tanh(first_value / 100.0)        # trying with some normalizing 
        obs[1] = first_value

        self.problem = prev_problem

        return obs, first_value














