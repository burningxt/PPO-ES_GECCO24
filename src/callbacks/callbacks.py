from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os


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
    def __init__(self, save_path: str, seed: int, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.seed = seed
        self.last_episode = 0

    def _on_step(self) -> bool:
        current_episode = self.training_env.envs[0].unwrapped.current_episode

        # Save if it's the first episode, then every 120 episodes
        # changed to 60 
        if current_episode == 1 or current_episode % (20) == 0:
            if current_episode != self.last_episode:
                self.last_episode = current_episode
                if self.verbose > 0:
                    print(f"Saving model at episode {current_episode}, seed {self.seed}")
                model_filename = f"model_seed_{self.seed}_episode_{current_episode}.zip"
                save_path = os.path.join(self.save_path, model_filename)
                os.makedirs(self.save_path, exist_ok=True)
                try:
                    self.model.save(save_path)
                except Exception as e:
                    print(f"Error saving model at episode {current_episode}: {e}")

        return True

# used to update the enavironment based on the info dict returned from the step each time
"""TODO move the helpers into another function its stupid to ahve this here"""
class UpdateEnvCallback(BaseCallback):
    def __init__(self, algo_name: str):
        super().__init__()
        # matching from the get_mean_q from SPACE
        self.algo_name = algo_name
        self.last_q = 0.0
        curriculum_size = 0

    # is called on step, but only triggers on a particular step 
    def _on_step(self) -> bool:
        # should be a list of dicts? 
        # episode_ended = self.locals.get("infos", None)
        
        # if the episode ended, then check for if the index has reached the end of the instance set
        # if episode_ended:
        current_index = self.training_env.envs[0].unwrapped.curriculum_index
        curriculum = self.training_env.envs[0].unwrapped.curriculum.curriculum
        self.curriculum_size = len(curriculum)

        # this means that the current curriculum has been exhausted, need to reset
        if current_index <= len(curriculum) - 1:

            
            # would want to update the policy explicitly here if not done already
            # to match the space code exactly, here is where I would consume the rollout buffer with train 
            self.update_curriculum_size(curriculum)
            self.update_curriculum()

        return True

    def update_curriculum_size(self, curriculum):
        mean_q = self.get_mean_q(curriculum)
        delta_q = np.abs(np.abs(mean_q) - np.abs(self.last_q))
        self.last_q = mean_q

        if (delta_q <= args.eta * np.abs(last_q) and len(curriculum) < total_instances):
            print("increasing instance set size")
            # just need to set it here, we use the curricluum list itself to keep track in the actual env


            """TODO: fix, it isn't supposed to just increase by 1 """
            temp = self.curriculum_size
            self.curriculum_size = temp + 1


    def update_curriculum(self):


    def get_mean_q(self, curriculum):
        # average value over instances
        """TODO does model in SB3 have a .value method
            don't think so, I think it instead might use predict_values instead?
        
        """
        qs = []
        # n_insts = self.model.env.env_method("get_instance_size")[0]
        n_insts = len(curriculum)
        """TODO does this call to self.model.env work, is this pointing to the actual env correctly?
            Will the reset method work correctly, will it return all the info I need?
        
        """
        env = self.model.env

        for i in range(n_insts):
            obs = env.reset()
            if self.algo_name == "trpo":
                # val = self.model.policy_pi.value([obs])
                val = self.model.predict_values(obs)
            # this is for PPO 
            elif self.algo_name == "ppo":
                # val = self.model.value(obs)
                val = self.model.predict_values(obs)
            else:
                print("Algo name not recognized")
            qs.append(val)
            # its over a flattened array 
        return np.mean(qs)


        



        # for i, info in enumerate(infos):
        #     if "episode" in info:
        #         ep_rew = infop"episode"





# class SaveOnEpisodes(BaseCallback):
#     def __init__(self, save_path: str, seed: int, verbose=1):
#         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
#         self.save_path = save_path
#         self.seed = seed
#         self.last_episode = 0

#     def _on_step(self) -> bool:
#         current_episode = self.training_env.envs[0].unwrapped.current_episode

#         # Save if it's the first episode, then every 120 episodes
#         if current_episode == 1 or current_episode % (12 * 10) == 0:
#             if current_episode != self.last_episode:
#                 self.last_episode = current_episode
#                 if self.verbose > 0:
#                     print(f"Saving model at episode {current_episode}, seed {self.seed}")
#                 model_filename = f"model_seed_{self.seed}_episode_{current_episode}.zip"
#                 save_path = os.path.join(self.save_path, model_filename)
#                 os.makedirs(self.save_path, exist_ok=True)
#                 try:
#                     self.model.save(save_path)
#                 except Exception as e:
#                     print(f"Error saving model at episode {current_episode}: {e}")

#         return True

