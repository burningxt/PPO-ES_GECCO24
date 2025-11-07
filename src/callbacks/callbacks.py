from stable_baselines3.common.callbacks import BaseCallback
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
        if current_episode == 1 or current_episode % (12 * 10) == 0:
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
