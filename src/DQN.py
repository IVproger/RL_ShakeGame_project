import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Metrics for plotting
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_values = []
        self.average_q_values = []

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), None
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state = state.reshape((1, -1))
        with torch.no_grad():
            # print(">>>>", state.shape)
            q_values = self.model(state)
        return torch.argmax(q_values).item(), None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        states = states.reshape((self.batch_size, -1))
        next_states = next_states.reshape((self.batch_size, -1))

        # Compute current Q-values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Update the Q-network
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log the loss
        self.episode_losses.append(loss.item())

        # Track average Q-value
        self.average_q_values.append(q_values.mean().item())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename):
        """Saves the entire agent to a file."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
            },
            'memory': list(self.memory),  # Convert deque to list
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
            epsilon=state['hyperparameters']['epsilon'],
            epsilon_decay=state['hyperparameters']['epsilon_decay'],
            epsilon_min=state['hyperparameters']['epsilon_min'],
            batch_size=state['hyperparameters']['batch_size'],
        )
        # Restore the agent's state
        agent.model.load_state_dict(state['model_state_dict'])
        agent.target_model.load_state_dict(state['target_model_state_dict'])
        agent.optimizer.load_state_dict(state['optimizer_state_dict'])
        agent.memory = deque(state['memory'], maxlen=len(state['memory']))
        print(f"Agent loaded from {filename}")
        return agent

    # Train the agent
    def train(self, env, episodes=1000, update_target_every=10, save_plots=False, plots_path='dqn_training_plots.png'):
        for episode in tqdm(range(episodes), desc="Training", unit='episode'):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action, _ = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                total_reward += reward

            self.episode_rewards.append(total_reward)
            self.epsilon_values.append(self.epsilon)

            if (episode + 1) % update_target_every == 0:
                self.update_target_model()

        if save_plots:
            self.save_plots(plots_path)
    
    def save_plots(self, plots_path):
        plots_dir = os.path.dirname(plots_path)
        os.makedirs(plots_dir, exist_ok=True)

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Rewards per episode
        axs[0, 0].plot(self.episode_rewards)
        axs[0, 0].set_title("Episode Rewards")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Total Reward")

        # Loss per episode
        axs[0, 1].plot(self.episode_losses)
        axs[0, 1].set_title("Loss Over Training")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Loss")

        # Epsilon decay
        axs[1, 0].plot(self.epsilon_values)
        axs[1, 0].set_title("Epsilon Decay")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Epsilon Value")

        # Average Q-values
        axs[1, 1].plot(self.average_q_values)
        axs[1, 1].set_title("Average Q-Values")
        axs[1, 1].set_xlabel("Episode")
        axs[1, 1].set_ylabel("Average Q-Value")

        plt.tight_layout()
        plt.savefig(plots_path)
        plt.close()