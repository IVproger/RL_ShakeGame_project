import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Metrics for plotting
        self.episode_rewards = []
        self.episode_losses = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
        action_probs = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def remember(self, log_prob, reward):
        if not hasattr(self, 'log_probs'):
            self.log_probs = []
        if not hasattr(self, 'rewards'):
            self.rewards = []
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        """Update the policy using stored rewards and log probabilities."""
        if not hasattr(self, 'log_probs') or len(self.log_probs) == 0:
            return

        returns = self.compute_returns(self.rewards)
        loss = -torch.sum(torch.stack(self.log_probs) * returns)  # Negative log-prob * return

        # Update the policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log the loss
        self.episode_losses.append(loss.item())

        # Clear memory
        self.log_probs.clear()
        self.rewards.clear()

    def compute_returns(self, rewards):
        """Compute discounted returns for an episode."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize returns to improve training stability
        if len(returns) > 1 and returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def save(self, filename):
        """Saves the entire agent to a file."""
        state = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
            },
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Agent saved to {filename}")

    @classmethod
    def load(cls, filename, lr=0.001):
        """Loads the agent from a file."""
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        # Recreate the agent
        agent = cls(
            state['hyperparameters']['state_dim'],
            state['hyperparameters']['action_dim'],
            lr=lr,
            gamma=state['hyperparameters']['gamma'],
        )
        # Restore the agent's state
        agent.policy.load_state_dict(state['policy_state_dict'])
        agent.optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f"Agent loaded from {filename}")
        return agent

    def train(self, env, episodes=1000, save_plots=False, plots_path='reinforce_training_plots.png'):
        for episode in tqdm(range(episodes), desc="Training", unit="episode"):
            state = env.reset()
            total_reward = 0
            done = False
            self.policy.train()

            while not done:
                action, log_prob = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(log_prob, reward)
                state = next_state
                total_reward += reward

            self.update_policy()
            self.episode_rewards.append(total_reward)

        if save_plots:
            self.save_plots(plots_path)
        self.policy.eval()


    def save_plots(self, plots_path):
        plots_dir = os.path.dirname(plots_path)
        os.makedirs(plots_dir, exist_ok=True)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Rewards per episode
        axs[0].plot(self.episode_rewards)
        axs[0].set_title("Episode Rewards")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Total Reward")

        # Loss per episode
        axs[1].plot(self.episode_losses)
        axs[1].set_title("Loss Over Training")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Loss")

        plt.tight_layout()
        plt.savefig(plots_path)
        plt.close()
