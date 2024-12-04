import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
from src.ParObsSnakeEnv import ParObsSnakeEnv
from src.PolicyNetwork import REINFORCEAgent
from src.utils import run_simulation

if __name__ == '__main__':
    env = ParObsSnakeEnv(grid_size=20, interact=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent.load('models/polNet/polNet_agent_par_10000_10.pkl')
    run_simulation(agent, env, num_simulations=1)