import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
from src.ParObsSnakeEnv import ParObsSnakeEnv
from src.Qlearning import QLearningAgent
from src.utils import run_simulation

if __name__ == '__main__':
    env = ParObsSnakeEnv(grid_size=20, interact=True)
    agent = QLearningAgent(env, epsilon=0.0001)
    agent.load_table('models/q-learning/q_learning_table_par_50000_10.pkl')
    run_simulation(agent, env, num_simulations=1)