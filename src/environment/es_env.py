import cocoex
from src.environment.vanilla_es import ES
from src.config.config import FES_MAX, STATE_SIZE, ACTION_SIZE, POP_SIZE, SIGMA_0, TRAIN_INSTANCE
from src.utilities.plotting import Draw
from src.utilities.tools import find_project_root
import gymnasium as gym
import numpy as np
import os
import copy


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
                 debug_logger=None,
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
        self.debug_logger = debug_logger

        # self.total_state_size = 2 + self.num_training_instances

        self.curriculum_size = 1
        self.curriculum_index = 0
        self.space_logger = space_logger


        # Initial Things Added
        self.use_space = use_space
        self.num_training_instances = num_training_instances

        # Setting the initial problem 
        self.problem = self.suite.get_problem(problem_index - 1)
        self.problem_index = problem_index
        self.before_first_rollout = True

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
# Should never be setting problem index
    def set_curriculum_index(self, curriculum_index: int):
        self.curriculum_index = curriculum_index
        self.problem = self.suite.get_problem(self.curriculum[curriculum_index-1]) # for the correct indesxing
        # self.space_logger.info("Set problem to: %d in (set_curriculum_index)", self.curriculum[curriculum_index-1])
        # self.space_logger.info("        Which corresponds to: %s", self.problem)


    def reset_curriculum_index(self):
        self.curriculum_index = 1 

    def set_curriculum(self, problem_list):
        # We're just setting this to be the problem list, whichis just [0] [1] [2] [3]...
        self.curriculum = problem_list
        # Always sets the curriculum index to be 

        # Dont Change
        self.current_index = 0
        # self.curriculum_index = 0
        self.problem = self.suite.get_problem(self.curriculum[0])

    def set_curriculum_size(self, size: int):
        self.curriculum_size = size

    # def next_problem_from_curriculum(self):
    # I belive this is also wrong anyway
    #     self.curriculum_index = (self.curriculum_index + 1) % len(self.curriculum)
    #     # Sets the problem index to whatever is held in the curriculum array 
    #     self.set_curriculum_index(self.curriculum[self.curriculum_index])

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

    def get_sigma(self):
        return self.es.sigma

    

    def step(self, action):
        success_ratio, improvement_ratio, norm_sigma, observation, reward = self._one_generation(action=action)

        # Added for potentially needed additional information
        infos = {}

        if self.mode == 'training':

            terminated = self.countevals >= self.fes_max # Why we have 41 (best evals) in each data file

            if terminated:
                # current switching system is accurate, but this number is one below what it really is

                self.space_logger.info(f"Completed episode %d, trained on %s", self.current_episode, self.problem)
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

                # Ordering for SPACE
                else:

                    # Gets the next problem from the current curriculum
                    if self.curriculum_index >= len(self.curriculum):
                        self.space_logger.info("ES environment curriculum empty, going to be changed soon...")
                        # self.space_logger.info(self.es.sigma)
                    else:

                        self.problem = self.suite.get_problem(self.curriculum[self.curriculum_index]) # don't need a -1 at the end because BBOB is already 0 indexed


                    # self.space_logger.info(f"--------------")
                    self.curriculum_index += 1



        elif self.mode == 'testing':

            
            self.debug_logger.debug(f"Executing test step inside es_env on problem index: %d ", self.problem_index)


        
            terminated = self.countevals >= self.fes_max + POP_SIZE * 2
        truncated = False

        return observation, reward, terminated, truncated, infos



        

    # Called from the Gym environment itself
    def _one_generation(self, action):

        # self.space_logger.info(f"Curriculm is currently %s", self.curriculum)

        solutions = self.es.ask()

        if action is None:
            # The default for if nothing is being changed by PPO
            scaling_factor = 1.0
        else:
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
        # Updated observation to include state

        # observation = np.array([norm_sigma,success_ratio, self.problem_index])
        
        # Converts to a one hot encoding to prevent inaccurate trend being learned based on problem index itself
        curr_problem_id = self.get_current_problem_id()
        current_problem_one_hot = self.get_problem_one_hot(curr_problem_id)

        observation = np.concatenate((
            np.array([norm_sigma, success_ratio], dtype=np.float32),
            current_problem_one_hot
        )).astype(np.float32)



        self.previous_best_fitness = self.current_best_fitness
        self.countevals += POP_SIZE # Track total number of evaluations

        return success_ratio, improvement_ratio, norm_sigma, observation, reward



# the env in space returns obs[observation].

# This is called automatically after every episode. 
    # original reset for PPO-ES
    def reset(self, **kwargs):

        # we are creating anew environment just for resteing
        # we call reset every time when we do the instance evaluations. 
        # or we call it before each experiment, for the NUM_RUNs, which is currently 25 in hte config 

        # self.space_logger.info(f"RESET IS BEING CALLED ----")
        self.es = ES(self.dim, SIGMA_0, POP_SIZE)
        self.fitness_values = None
        self.current_best_fitness = None
        self.previous_best_fitness = None
        self.countevals = 0
        self.cumulative_reward = 0
        self.improvement_ratios = []
        # are there any other curriculum values I need to add
        # I don't really think so, out of hte new params I added all seem fine
        # Even stuff about the problem or fitness_values, like dim fes_max etc don't really bmatter

        ## Data Handling, evaluating the very first solution manually
        first_solution = self.es.xmean.copy()       # this is the "center" at generation 0
        # the xmean is the object used for the solutions
        first_value = self.problem(first_solution)

        curr_prob_id = self.get_current_problem_id()
        problem_one_hot = self.get_problem_one_hot(curr_prob_id)

        # obs = np.zeros(STATE_SIZE)
        obs = np.concatenate((
           np.array([0.0, 0.0], dtype=np.float32), 
           problem_one_hot
        )).astype(np.float32)
        # thgis should be the correct problem number, its not minus 1, so index 1 will be problem 1
        # self.curriculum_index = 0

        # reset is called every time an episode is complete anyways
            # for my implementation,e very episode is trained on one instance. 
        # If its above anyway, then this will be null. somewhere else is responsible for it 

        # Does reset do anything with the obs when returned after every episode 
        # This gives an out of index error because curriculum index is too 

        # Only doing this if its valid, if its not then its not going to use this anyway
        # Is -1 because, it will have just finished after an episode which would have incremented it one more time before its able to be reset 
        # if self.curriculum_index-1 < len(self.curriculum):
        #     problem = self.curriculum[self.curriculum_index-1]
        # else: 
        #     # gets the first entry, this is for usage 
        #     problem = self.curriculum[0]
        
        # obs[2] = problem

        # if self.space_logger:
        #     self.space_logger.info(f"In reset returning: {obs[2]}")

        return obs, first_value
    # , first_value

    
    # only trying with the improvement ratio and the sigma
    def poll_env(self, **kwargs):

        # we are creating anew environment just for resteing
        # we call reset every time when we do the instance evaluations. 
        # or we call it before each experiment, for the NUM_RUNs, which is currently 25 in hte config 

        # self.space_logger.info(f"Old ES Sigma is: ")
        # self.space_logger.info(self.es.sigma)

        # old_state = self.save_state()


        # setting sigma back as part of the environment
        # self.es.sigma = old_state["es"].sigma

        # self.space_logger.info(f"Old ES Sigma  in save is: ")
        # self.space_logger.info(self.es.sigma)
        # are there any other curriculum values I need to add
        # I don't really think so, out of hte new params I added all seem fine
        # Even stuff about the problem or fitness_values, like dim fes_max etc don't really bmatter

        ## Data Handling, evaluating the very first solution manually
        first_solution = self.es.xmean.copy()       # this is the "center" at generation 0
        # the xmean is the object used for the solutions

        # Is just always here to be used as a baseline, to ensure hte success ratio is different 
        first_value = self.problem(first_solution)
        # This will always be the same, as it is different for each instance. 
        # is this going to be different as its part of the context. I guess we're measuring the improvement vs the last one, so its fine
        self.previous_best_fitness = first_value


        self.space_logger.info("First value on problem: ")

        self.space_logger.info(self.problem)
        self.space_logger.info(first_value)

        success_ratio, improvement_ratio, norm_sigma, observation, reward = self._one_generation(action=None)
        temp = []
        temp.append(norm_sigma)
        temp.append(success_ratio)

        # Taking the first array in the returned tuple, which is just the observations
        # obs = self.reset()[0] 
        # obs[0] = temp[0]
        # obs[1] = temp[1]
        # state[1] = self.get_reward()
        # Printing confirmed its just all zeros
        # Is no longer just all 0s, becuase its now after one generation
        # self.space_logger.info(f"First solution: ", first_solution)

        # TODO Therese a bug because the success ratio is always 0.2 which is the default because there is no previous generation
        
        return temp 


    def get_current_problem_id(self):
        if self.use_space: 
            if self.curriculum_index - 1 < len(self.curriculum) and self.curriculum_index - 1 >= 0:
                return self.curriculum[self.curriculum_index - 1]
            return self.curriculum[0]
        else:
            return self.problem_index - 1

    def get_problem_one_hot(self, problem_idx: int):

        one_hot = np.zeros(self.num_training_instances, dtype=np.float32)

        # Guard just in case 
        if 0 <= problem_idx < self.num_training_instances:
            one_hot[problem_idx] = 1.0
        return one_hot