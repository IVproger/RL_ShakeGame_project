import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

class SarsaAgent:
    def __init__(self, env, learning_rate=0.5, discount_factor=0.99, epsilon=0.1, learning_rate_decay=0.8, epsilon_decay=0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate_decay = learning_rate_decay
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def choose_action(self, state):
        state = tuple(state.flatten())
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample(), None  # Explore
        else:
            return np.argmax(self.q_table[state]), None  # Exploit

    def update_q_value(self, state, action, reward, next_state, next_action):
        next_state = tuple(next_state.flatten())
        state = tuple(state.flatten())
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def save_table(self, table_path):
        with open(table_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_table(self, table_path):
        with open(table_path, 'rb') as f:
            self.q_table = pickle.load(f)

    def train(self, num_episodes, save_plots=False, plots_path='plots.png'):
        self.episode_rewards = []
        self.steps_per_episode = []
        self.epsilon_values = []
        self.learning_rate_values = []
        self.average_q_updates = []

        for episode in tqdm(range(num_episodes), desc='Training', unit='Episode'):
            state = self.env.reset()
            action, _ = self.choose_action(state)
            done = False
            total_reward = 0
            steps = 0
            q_update_magnitudes = []

            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action, _ = self.choose_action(next_state)

                old_q_value = self.q_table[tuple(state.flatten())][action]
                self.update_q_value(state, action, reward, next_state, next_action)
                new_q_value = self.q_table[tuple(state.flatten())][action]
                q_update_magnitudes.append(abs(new_q_value - old_q_value))

                total_reward += reward
                state = next_state
                action = next_action
                steps += 1

            # Logging metrics for the episode
            self.episode_rewards.append(total_reward)
            self.steps_per_episode.append(steps)
            self.epsilon_values.append(self.epsilon)
            self.learning_rate_values.append(self.learning_rate)
            self.average_q_updates.append(np.mean(q_update_magnitudes))

            # Decay epsilon and learning rate periodically
            if episode % 1000 == 0 and episode != 0:
                self.epsilon *= self.epsilon_decay
                self.learning_rate *= self.learning_rate_decay

        # Save plots after training
        if save_plots:
            self.save_plots(plots_path)

    def save_plots(self, plots_path):
        plots_dir = os.path.dirname(plots_path)
        os.makedirs(plots_dir, exist_ok=True)

        # Create a figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # Plot rewards
        axs[0, 0].plot(self.episode_rewards)
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Cumulative Reward')
        axs[0, 0].set_title('Rewards Over Episodes')

        # Plot steps per episode
        axs[0, 1].plot(self.steps_per_episode)
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Steps per Episode')
        axs[0, 1].set_title('Steps Over Episodes')

        # Plot epsilon values
        axs[1, 0].plot(self.epsilon_values)
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Epsilon')
        axs[1, 0].set_title('Epsilon Decay Over Episodes')

        # Plot learning rate values
        axs[1, 1].plot(self.learning_rate_values)
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Learning Rate')
        axs[1, 1].set_title('Learning Rate Decay Over Episodes')

        # Plot Q-value updates
        axs[2, 0].plot(self.average_q_updates)
        axs[2, 0].set_xlabel('Episode')
        axs[2, 0].set_ylabel('Average Q-Value Update')
        axs[2, 0].set_title('Q-Value Updates Over Episodes')

        # Hide the empty subplot (bottom right)
        fig.delaxes(axs[2, 1])

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(plots_path)
        plt.close()