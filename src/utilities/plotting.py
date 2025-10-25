import numpy as np
import os
from src.config.config import POP_SIZE, EPISODES
from src.utilities.tools import load_data
from scipy.stats import sem, rankdata
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


class Draw:
    @staticmethod
    def plot_loss_data(base_dir, loss_history):
        results_dir = os.path.join(base_dir, 'output_data', 'results')
        plt.plot(loss_history)
        plt.title('Loss over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.tight_layout()  # Adjust the layout
        # plt.savefig(os.path.join(results_dir, 'loss_data.png'), dpi=1200)  # Save the plot
        plt.show()  # Display the plot

    @staticmethod
    def plot_episode_data(base_dir, episode_data, problem_index):
        """
        Plots [episode, best_fitness_value] and [episode, total_reward] in two subplots.

        :param episode_data: List of [episode, best_fitness_value, total_reward].
        :param save_path: Path to save the plot.
        """
        results_dir = os.path.join(base_dir, 'output_data', 'results')
        # Convert the input output_data to numpy arrays for easier handling
        episode_data = np.array(episode_data)
        episodes = episode_data[:, 0]
        best_fitness_values = episode_data[:, 1]
        total_rewards = episode_data[:, 2]

        # Initialize the plot
        plt.figure(figsize=(8, 6))

        # Subplot for best fitness value
        plt.subplot(2, 1, 1)
        plt.plot(episodes, best_fitness_values, marker='o', color='cadetblue')
        plt.yscale('symlog')  # Set the y-axis to a logarithmic scale
        plt.xlabel('Episode')
        plt.ylabel('Best Fitness Value (log scale)')
        plt.title('Best Fitness Value per Episode (Logarithmic Scale) PPO')

        # Subplot for total reward
        plt.subplot(2, 1, 2)
        plt.plot(episodes, total_rewards, marker='o', color='darksalmon')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward per Episode')

        plt.tight_layout()  # Adjust the layout
        plt.savefig(os.path.join(results_dir, f'episode_data_problem_{problem_index}.pdf'), dpi=1200)  # Save the plot

    @staticmethod
    def plot_convergence_data_median_variance(problem_index, episodes_tested_dir, baselines_dir, plot_dir, instance):

   
        plt.figure(figsize=(10, 6))
        plt.title(f'Convergence Comparison Across Episodes, CMA-ES, and One-Fifth ES on Problem F{problem_index}')
        plt.yscale('symlog')
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Best Fitness Value (log scale)')
        plt.grid(True)

        # Define a list of colors for the episodes, ensure there are enough distinct colors for all elements
        episode_colors = ['steelblue', 'sandybrown', 'firebrick']

        for i, episode in enumerate(EPISODES):
            fitness_values = load_data(
                os.path.join(episodes_tested_dir, f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'))
            median_fitness_values = np.mean(fitness_values, axis=0)
            min_fitness_values = np.min(fitness_values, axis=0)
            max_fitness_values = np.max(fitness_values, axis=0)
            evaluations = np.arange(len(median_fitness_values)) * POP_SIZE
            color = episode_colors[i % len(episode_colors)]  # Use distinct colors for each episode
            if episode == 1200:
                plt.plot(evaluations, median_fitness_values, label=f'Episode {episode}', color=color, linewidth=2)
            else:
                plt.plot(evaluations, median_fitness_values, label=f'Episode {episode}', color=color)
            plt.fill_between(evaluations, min_fitness_values, max_fitness_values, color=color, alpha=0.2)

        # Plotting for CMA-ES
        fitness_values_cma_es = load_data(
            os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
        median_cma_es_fitness = np.median(fitness_values_cma_es, axis=0)
        max_cma_es_fitness = np.max(fitness_values_cma_es, axis=0)
        min_cma_es_fitness = np.min(fitness_values_cma_es, axis=0)
        evaluations = np.arange(len(median_cma_es_fitness)) * POP_SIZE
        plt.plot(evaluations, median_cma_es_fitness, label='CMA-ES', color='black', linewidth=2)
        plt.fill_between(evaluations, min_cma_es_fitness, max_cma_es_fitness, color='grey', alpha=0.2)

        # Plotting for One-Fifth ES
        fitness_values_one_fifth_es = load_data(
            os.path.join(baselines_dir, f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'))
        median_one_fifth_es_fitness = np.median(fitness_values_one_fifth_es, axis=0)
        max_one_fifth_es_fitness = np.max(fitness_values_one_fifth_es, axis=0)
        min_one_fifth_es_fitness = np.min(fitness_values_one_fifth_es, axis=0)
        evaluations = np.arange(len(median_one_fifth_es_fitness)) * POP_SIZE
        plt.plot(evaluations, median_one_fifth_es_fitness, label='One-Fifth ES', color='darkseagreen', linewidth=2)
        plt.fill_between(evaluations, min_one_fifth_es_fitness, max_one_fifth_es_fitness, color='darkseagreen', alpha=0.2)

        plot_filename = os.path.join(plot_dir, f'convergence_comparison_plot_F{problem_index}.pdf')
        plt.legend()
        plt.savefig(plot_filename, dpi=1200, bbox_inches='tight')
        plt.close()

    @staticmethod
    # FLAGVP this is what makes the plots ive been looking at 
    def plot_convergence_data_mean_ci(problem_index, episodes_tested_dir, baselines_dir, plot_dir, instance):
        plt.figure(figsize=(10, 6))
        plt.title(f'Convergence Comparison Across Episodes, CMA-ES, and One-Fifth ES on Problem F{problem_index}')
        plt.yscale('symlog')
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Fitness Value (Mean with 95% CI)')
        plt.grid(True)


        if problem_index == 1:
            print(
                "problem_index: ", problem_index,
                "episodes_testted_dir: ", episodes_tested_dir,
                "baselines_dir: ", baselines_dir,
                "plot dir: ", plot_dir, 
                "instances: ", instance
            )
        # else: 
        #     print(
        #         "problem_index: ", problem_index,
        #         "episodes_testted_dir: ", episodes_tested_dir,
        #         "baselines_dir: ", baselines_dir,
        #         "plot dir: ", plot_dir, 
        #         "instances: ", instance
        #     )

        episode_colors = ['steelblue', 'sandybrown', 'firebrick']

        for i, episode in enumerate(EPISODES):
            fitness_values = load_data(
                os.path.join(episodes_tested_dir, f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'))
            mean_fitness_values = np.mean(fitness_values, axis=0)
            ci = 1.96 * sem(fitness_values, axis=0)  # 95% CI
            evaluations = np.arange(len(mean_fitness_values)) * POP_SIZE
            color = episode_colors[i % len(episode_colors)]
            plt.plot(evaluations, mean_fitness_values, label=f'Episode {episode}', color=color, linewidth=2)
            plt.fill_between(evaluations, mean_fitness_values - ci, mean_fitness_values + ci, color=color, alpha=0.2)

        # Plotting for CMA-ES
        fitness_values_cma_es = load_data(os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
        mean_cma_es_fitness = np.mean(fitness_values_cma_es, axis=0)
        ci_cma_es = 1.96 * sem(fitness_values_cma_es, axis=0)
        evaluations = np.arange(len(mean_cma_es_fitness)) * POP_SIZE
        plt.plot(evaluations, mean_cma_es_fitness, label='CMA-ES', color='black', linewidth=2)
        plt.fill_between(evaluations, mean_cma_es_fitness - ci_cma_es, mean_cma_es_fitness + ci_cma_es, color='grey',
                         alpha=0.2)

        # Plotting for One-Fifth ES
        fitness_values_one_fifth_es = load_data(
            os.path.join(baselines_dir, f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'))
        mean_one_fifth_es_fitness = np.mean(fitness_values_one_fifth_es, axis=0)
        ci_one_fifth_es = 1.96 * sem(fitness_values_one_fifth_es, axis=0)
        evaluations = np.arange(len(mean_one_fifth_es_fitness)) * POP_SIZE
        plt.plot(evaluations, mean_one_fifth_es_fitness, label='One-Fifth ES', color='darkseagreen', linewidth=2)
        plt.fill_between(evaluations, mean_one_fifth_es_fitness - ci_one_fifth_es,
                         mean_one_fifth_es_fitness + ci_one_fifth_es, color='darkseagreen', alpha=0.2)

        plot_filename = os.path.join(plot_dir, f'convergence_comparison_plot_F{problem_index}.pdf')
        plt.legend()
        plt.savefig(plot_filename, dpi=1200, bbox_inches='tight')
        plt.close()

    @staticmethod
    def standardize_data(values):
        mean_val = np.mean(values)
        std_val = np.std(values)
        # print(mean_val)
        # print(std_val)
        # one of them has a 0 as a std_val, which causes the invalid warning
        if std_val == 0:
            std_val += .0001
        standardized = (values - mean_val) / std_val
        # print("standerdized is: ", standardized)
        return standardized

    def plot_standardized_performance_boxplot(self, episodes_tested_dir, baselines_dir, plot_dir, instance):
        all_fitness_values = {'Episode 1': [], 'Episode 600': [], 'Episode 1200': [], 'CMA-ES': [], 'One-Fifth ES': []}

        # Collect and standardize data for each EP
        for episode in EPISODES:  # EPISODES should be something like [1, 600, 1200]
            key = f'Episode {episode}'
            for problem_index in range(1, 25):  # Assuming 24 problems
                fitness_values = load_data(
                    os.path.join(episodes_tested_dir, f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'))
                standardized_values = self.standardize_data(fitness_values[:, -1])
                all_fitness_values[key].extend(standardized_values)

        # Collect and standardize data for CMA-ES
        for problem_index in range(1, 25):
            fitness_values = load_data(os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
            standardized_values = self.standardize_data(fitness_values[:, -1])
            all_fitness_values['CMA-ES'].extend(standardized_values)

        # Collect and standardize data for One-Fifth ES
        for problem_index in range(1, 25):
            fitness_values = load_data(os.path.join(baselines_dir, f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'))
            standardized_values = self.standardize_data(fitness_values[:, -1])
            all_fitness_values['One-Fifth ES'].extend(standardized_values)

        # Plotting the box plot with standardized data
        plt.figure(figsize=(12, 8))
        plt.title('Standardized Overall Performance Comparison')
        plt.ylabel('Standardized Final Fitness Value (z-score)')
        plt.boxplot(all_fitness_values.values(), labels=all_fitness_values.keys())
        plt.grid(True)

        plot_filename = os.path.join(plot_dir, 'standardized_overall_performance_comparison_boxplot.pdf')
        plt.savefig(plot_filename, dpi=1200, bbox_inches='tight')
        plt.close()


