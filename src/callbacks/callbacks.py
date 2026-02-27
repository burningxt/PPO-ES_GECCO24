from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3.common.utils import obs_as_tensor
import torch as th
import logging  


class LearningRateScheduler(BaseCallback):
    def __init__(self, initial_learning_rate, scheduler, verbose=0):
        super(LearningRateScheduler, self).__init__(verbose)
        self.scheduler = scheduler
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = initial_learning_rate
        self.total_timesteps = 0  # Initialize the total_timesteps attribute

    def _on_training_start(self):
        # Update the optimizer's learning rate at the start of training
        self.update_learning_rate(self.initial_learning_rate)

    def _on_step(self):
        # Get the current progress remaining (from 1 to 0)
        progress_remaining = 1 - self.num_timesteps / self.total_timesteps
        # Calculate the current learning rate based on the progress
        self.current_learning_rate = self.scheduler(progress_remaining)
        # Update the optimizer's learning rate
        self.update_learning_rate(self.current_learning_rate)
        return True

    def update_learning_rate(self, new_learning_rate):
        # Set the new learning rate to the optimizer
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_learning_rate


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, save_path: str, seed: int, verbose=1, space_logger=None):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.seed = seed
        self.last_episode = 0
        self.space_logger = space_logger

    def _on_step(self) -> bool:
        # print("in save on best training reward callback")
        current_episode = self.training_env.envs[0].unwrapped.current_episode

        # Save if it's the first episode, then every 120 episodes
        # changed to 60 
        # if current_episode == 1 or current_episode % (20) == 0:

        # saves it every 100 episodes
        # if current_episode == 1 or current_episode % (100) == 0:
        if current_episode == 1 or current_episode % (60) == 0:
            if current_episode != self.last_episode:
                self.last_episode = current_episode
                if self.verbose > 0:
                    print(f"Saving model at episode {current_episode}, seed {self.seed}")
                model_filename = f"model_seed_{self.seed}_episode_{current_episode}.zip"
                save_path = os.path.join(self.save_path, model_filename)
                print("saving at: ", save_path)
                self.space_logger.modellog.info(f"Saved model seed=%d episode=%d path=%s", self.seed, current_episode, save_path)
                os.makedirs(self.save_path, exist_ok=True)
                try:
                    self.model.save(save_path)
                except Exception as e:
                    print(f"Error saving model at episode {current_episode}: {e}")

        return True

# Used to update the environment curriculum at each step 
class UpdateEnvCallback(BaseCallback):
    def __init__(self, algo_name: str, space_logger, use_space=1):
        super().__init__()
        self.algo_name = algo_name
        self.last_q = 0.0
        self.curriculum_size = 1
        self.curriculum = []
        self.space_logger = space_logger

        # For now just training on 12 total instances
        self.num_training_instances = 12
        self.use_space=use_space
        self.update_counter=0



    def _on_training_start(self):
        """
        At the start of training, ensure that curriculum is already set.
        """
        # Just do nothing
        if not self.use_space:
            return True

        # On training start, initialize the curriculum
        self.update_curriculum()

    def _on_step(self) -> bool:

        """
        Runs after each call to "step()" function in es_env.py. Checks each time to see if full curriculum
        has been exhausted, and if it has then it may increase curriculum size and will set the next curriculum. 
        
        :param self: Description
        :return: If the function completed successfully
        """


        # a rollout is collected every 12 * 400 steps, which is n_steps in the ppo_es_model file
        if not self.use_space:
            # Just don't do anything if space isn't being used
            return True

        # Retrieve the current index and curriculum
        current_index = self.training_env.envs[0].unwrapped.curriculum_index
        curriculum = self.training_env.envs[0].unwrapped.curriculum

        # This means that the current curriculum has been exhausted, need to reset
        if current_index > len(curriculum):
            self.space_logger.info("Current Curriculum exhausted, transitioning...")

            sigma_val = self.training_env.envs[0].unwrapped.es.sigma

            self.space_logger.info(sigma_val)

            self.space_logger.info("Second Sigma Val:")


            self.space_logger.info(self.training_env.envs[0].unwrapped.get_sigma())


            # Update the curriculum size
            self.update_curriculum_size(curriculum)

            # Update the curriculum itself
            self.update_curriculum()

        return True

    def _on_rollout_end(self):
        self.update_counter += 1
        # our total timesteps is set to 12 * 4000, it keeps training until that is done right now. I think this is what actually stops the training. 
        # each timestep is just one call to the step.env, which is done within the env_es
        # each rollout is 12 * 400 steps
        self.space_logger.info(f"Collected rollout:")
        self.space_logger.info(f"                  about to do update [%d] to the policy", self.update_counter)
        self.space_logger.info(f"                  at [%d] model timesteps so far", self.num_timesteps)
        return True
        # return super()._on_rollout_end()

    def update_curriculum_size(self, curriculum):
        """
        Helper function used to update the size of the curriculum

        If learning sufficiently converged, we can add more instances. 
        
        """
        STEP_SIZE_CONST = 1

        # Calculate mean_q
        eval_env = self.training_env.envs[0].unwrapped 

        mean_q = self.get_mean_q(self.model, eval_env, curriculum)
        delta_q = np.abs(np.abs(mean_q) - np.abs(self.last_q))
        self.last_q = mean_q

        # SPACE had it defined as this
        eta_const = .1

        # If condition passes, then increase size by 1
        if (delta_q <= eta_const * np.abs(self.last_q) and len(curriculum) < self.num_training_instances):
            temp = self.curriculum_size
            self.curriculum_size = temp + STEP_SIZE_CONST
            self.space_logger.info(f"Updating size from: [%d] to [%d]", temp, self.curriculum_size)


    def update_curriculum(self):
        """
        Helper function used to update the curriculum
        """
        NUM_FUNCTIONS  = 12 # TODO make this not hard coded

        # Set the env
        eval_env = self.training_env.envs[0].unwrapped 
        temp = self.order_instances_qvals(self.model, eval_env, self.num_training_instances)
        self.curriculum = temp
        new_curriculum = self.curriculum[:self.curriculum_size]
        
        # Set the updated size, and the updated curriculum
        self.training_env.envs[0].unwrapped.set_curriculum_size(self.curriculum_size)
        self.training_env.envs[0].unwrapped.set_curriculum(new_curriculum)
        self.training_env.envs[0].unwrapped.reset_curriculum_index()
        self.space_logger.info(f"Updating curriculum to: %s", new_curriculum)

        # This is so that it ignores the first one, which is already set as callbacks run after the step
        self.training_env.envs[0].unwrapped.set_curriculum_index(1) 
    

    # Returns indices in ascending order, used for "absolute"
    def order_instances_qvals(self,learner, env, num_instances):
        # Order the instances by q value
        evals = self.get_instance_evals(learner, env, num_instances)
        return np.argsort(evals)


    # Returns a numpy array of value estimates 
    # This is the one used for ordering the instances. 
    def get_instance_evals(self,learner, env, num_instances):

        # Retrieve the instance evaluations for every instance being trained on
        evals = []
        prev_set = env.get_curriculum()
        obs_list = []

        for i in range(num_instances):

            # When you set the curriculum in the environment, it correctly sets its own problem as the first item in that curriclumu
            env.set_curriculum([i])
            # Rest and go to starting state for this instance
            # obs, info = env.reset()
            obs, info = env.poll_env()
                
            # obs, info = env.env_method("poll_env", indices=0)[0] # indices refer to the environment, we only hav e1

            self.space_logger.info(f"Collected observation for instance: [%d]", i)
            self.space_logger.info(obs)

            # TODO not entirely sure
            # trying to fix shape mismatch
            # I kept getting dimension out of range errors 
            obs_t = obs_as_tensor(obs, learner.device)
            # if obs_t.ndim == 1:
            # This is needed to fix the out of range eror 
            obs_t = obs_t.unsqueeze(0)
            obs_list.append(obs_t)

            # else:

                # print("THIS IS BEING TRIGGERED ITS STILL NEEDED")
            val = 0

            if self.algo_name == "trpo":
                # value is the network's estimate of the expected discounted return from that initial state 
                # val = learner.policy_pi.value([obs_as_tensor(obs, self.model.device)])
                val = learner.policy_pi.policy.predict_values([obs_as_tensor(obs, self.model.device)])
            else:
                # val = learner.policy.predict_values(obs_as_tensor(obs[0], self.model.device))
                val_t = learner.policy.predict_values(obs_t)
                val = float(val_t.detach().cpu().numpy().squeeze())

            evals.append(val)

        # set the environment back to what it was before
        env.set_curriculum(prev_set)

        self.space_logger.info("Collected instance evals: %s", evals)

        self.space_logger.info("    And obs: %s", obs_list)
        return np.array(evals)



    def get_mean_q(self, learner, eval_env, curriculum):
        """
        Get the mean_q for every instance, and return the mean. 

        Adjusted from the SPACE implementation, to account for differences with SB1 and SB3
        """
        qs = []
        n_insts = len(curriculum)
        # env = self.model.env

        env = self.training_env.envs[0].unwrapped 
        # env = self.training_env


        for i in range(n_insts):
            # obs = env.reset()
            obs, first_val  = env.poll_env()
            # obs, info = env.env_method("poll_env", indices=0)[0]
            val = 0
            obs_t = obs_as_tensor(obs, learner.device)
            # if obs_t.ndim == 1:
            # This is needed to fix the out of range eror 
            obs_t = obs_t.unsqueeze(0)
            # obs_list.append(obs_t)

            if self.algo_name == "trpo":
                # val = self.model.policy_pi.value([obs])
                # I made it not have gradients since its evaluating not training here
                # predict_values Get the estimated values according to the current policy given the observations.
                with th.no_grad():
                    val = self.model.policy.predict_values(obs_as_tensor(obs, self.model.device))
                val = val.cpu().numpy()
            # this is for PPO 
            elif self.algo_name == "ppo":
                # val = self.model.value(obs)
                # with th.no_grad():
                    # val = self.model.policy.predict_values(obs_as_tensor(obs, self.model.device))
                val_t = learner.policy.predict_values(obs_t)
                val = float(val_t.detach().cpu().numpy().squeeze())


                # val = val.cpu().numpy()
            else:
                print("Algo name not recognized")
            qs.append(val)
            # its over a flattened array 
        return np.mean(qs)


        


