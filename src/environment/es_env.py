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
    """ # FLAGVP I think this is actually the environment for PPO-ES I think 

    def __init__(self,
                 problem_type="bbob",
                 instance=1,
                 dim=40,
                 fes_max=FES_MAX,
                 sigma_0=SIGMA_0,
                 problem_index=1,
                 seed=None):
        super(ES_Env, self).__init__() # inheriting the constructor 
        self.seed(seed)  # Set the seed using the inherited method
                                  # the problem type is just bbob by default 

        # create a new cocoex suite
        self.suite = cocoex.Suite(problem_type,
                             "",# empty string for the benchmark options (we don't use any)  # instance is 1 by default
                             f"dimensions: {dim} function_indices:1-24 instance_indices:{instance}")
        # select which problem from the suite, by index (0 based indexing is used internally)
        # this is likely what i'll have to change to improve on space, getting the next problem 
        # it just trains on problem

        # starting with the one provided 
        self.curriculum = [problem_index]
        # can either remove them from the list, but may want that information later so will index for now
        # it is ordered
        self.curriculum_index = 0
        # default value
        self.curriculum_size = 1


        # 
        self.problem = self.suite.get_problem(problem_index - 1)
        self.problem_index = problem_index
        # self.problem_optimum_value = self.problem.final_target_f
        # #print("*************** abvout to print optimal value***********")
        # #print("GLOBAL OPTIMA OF ", problem_index, " is : ", self.problem_optimum_value)


        # we make the ES here 
        # this is what the initial solution will be based on 
        # sigma 0 is 0.5
        # so the first evaluated population is POP_SIZE 25 samples drawn from a normal distribution sampled at sigma_0 which is .5
        self.es = ES(dim, sigma_0, POP_SIZE)

        # 
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

    def set_problem_index(self, problem_index: int):
        self.problem_index = problem_index
        self.problem = self.suite.get_problem(self.problem_index - 1) # for the correct indesxing

    def set_curriculum(self, problem_list):
        self.curriculum = problem_list
        self.current_index = 0
        # always does the first entry in the curriculum at the start. 
        # this helps with the instance estimates as well 
        # the actual problem index in cocoex is just 1 higher than what it is in the curriculum
        self.problem = self.suite.get_problem(self.curriculum[0])
        print("problem in environment set to: ")
        print(self.problem)

    def set_curriculum_size(self, size: int):
        self.curriculum_size = size

    def next_problem_from_curriculum(self):
        self.curriculum_index = (self.curriculum_index + 1) % len(self.curriculum)
        # sets the problem index to whatever is held in the curriculum array 
        self.set_problem_index(self.curriculum[self.curriculum_index])

    def get_curriculum(self):

        return self.curriculum.copy()


    

    def seed(self, seed=None): # 
        np.random.seed(seed)

    def set_mode(self, mode):
        self.mode = mode

    def evaluate_fitness(self, solution):
        # just plug the solution into the problem
        # because once you plug the solution into the problem, it'll give you the fitness 
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

    # this is what regulates the countevals variable
    # I think this is whats used for PPO-ES, and nothing else?
    # I believe it is called from gym itself, which is why this is annoying to find
    def step(self, action):
        # self.es.ask is what returns the candidate solutions
        print("CURRENTLY IN STEP, TRAINING ON INSATNCE: ")
        print(self.problem)
        #print(self.curriculum)
        solutions = self.es.ask()
        scaling_factor = action[0] * 0.75 + 1.25
        self.es.sigma *= scaling_factor
        self.es.sigma = max(1e-20, min(100.0, self.es.sigma))
        # this is where the evaluation takes place, calculates it for all the the solutions returned by ask 
        # just calculate the fitness values for 
        self.fitness_values = np.apply_along_axis(self.problem, 1, solutions)
            # each row is a candidate solution vector (so a point in the search space) 
                # it evaluates the candidate solutions on the actual problem
                # this will return a single number, and the lower the better
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
        # This is the exact line that iterates countevals
        # pop size is set to 25
        # this is then used for both training and testing
        self.countevals += POP_SIZE
        infos = {}

        if self.mode == 'training':
                                        # because of this >= is why we have 41 in each data file
            terminated = self.countevals >= self.fes_max
            episode_end = False
            # this is used to determine when the current episode ends 
            if terminated:
                episode_end = True
                # we increment the episodes here 
                self.current_episode += 1
                self.episode_data.append([self.current_episode,
                                          self.current_best_fitness,
                                          self.cumulative_reward])

                self.problem = self.suite.get_problem(self.curriculum[self.curriculum_index-1])
                self.curriculum_index += 1


           
                # self.next_problem_from_curriculum()

                # if self.current_episode % 1200 == 0:
                #     Draw().plot_episode_data(self.base_dir, self.episode_data, self.problem_index)

            # don't need to pass in the self.curruculum because that's already saved I think
            # the infos is passed every time to detemrine when the callback should be triggered
            infos = {"episode_end": episode_end}


        elif self.mode == 'testing':
            # here is where it is for testing
            terminated = self.countevals >= self.fes_max + POP_SIZE * 2
        truncated = False

        return observation, reward, terminated, truncated, infos

    # def space_predictor(self) -> List(int):


    #     new_inst_num = 0
    #     #print("dummy")

        # this is going to return a list of the entires that we want to train on in the next episode 




        
        
    # def get_mean_q(self, model, algo):
    #     qs = []
    #     n_insts = model.env.env_method("get_instance_size")[0]
    #     env = model.env
    #     for i in range(n_insts):
    #         obs = env.reset()
    #         if algo == "trpo":
    #             val = model.policy_pi.value([obs])
    #         # this is for PPO 
    #         else:
    #             val = model.value(obs)
    #         qs.append(val)
    #         # its over a flattened array 
    #     return np.mean(qs)

    


# the env in space returns obs[observation]

    # reset for PPO-ES
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

        # choose current problem from curriculum
        # pid = self.curriculum[self.curriculum_index]
        # self.set_problem_index(pid)
        self.problem = self.suite.get_problem(self.curriculum[0])
        first_solution = self.es.xmean.copy()
        first_value = float(self.problem(first_solution))

        # observation should not be constant across problems
        obs = np.zeros(STATE_SIZE, dtype=np.float32)
        obs[0] = self.norm_input()                   # sigma-related
        obs[1] = np.tanh(first_value / 100.0)        # some scaled signal about problem difficulty

        # return obs, {"first_value": first_value, "problem_index": int(pid)}
        return obs, first_value














