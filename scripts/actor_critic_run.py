import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
from src.ParObsSnakeEnv import ParObsSnakeEnv
from src.ActorCritic import Actor
from src.utils import run_simulation

if __name__ == '__main__':
    env = ParObsSnakeEnv(grid_size=20, interact=True)
    num_actions = env.action_space.n
    num_inputs = env.observation_space.shape[0] 
    actor = Actor(num_inputs, num_actions)
    actor.load_state_dict(torch.load('models/actor-critic/actor_v1.pth'))
    run_simulation(actor, env, num_simulations=1)