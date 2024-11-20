import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(0, 1, (grid_size, grid_size, 3), dtype=np.float32)
        self.reset()
        self.seed()
    
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]  # Initial snake position
        self.food = self._place_food()
        self.direction = 1  # Initial direction: Right
        self.done = False
        return self._get_observation()

    def _place_food(self):
        while True:
            food_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food_pos not in self.snake:
                return food_pos

    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for x, y in self.snake:
            obs[x, y] = [0, 1, 0]  # Snake body in green
        obs[self.food[0], self.food[1]] = [1, 0, 0]  # Food in red
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called after environment is done")

        # Update direction
        if action == 0 and self.direction != 2:  # Up
            self.direction = 0
        elif action == 1 and self.direction != 3:  # Right
            self.direction = 1
        elif action == 2 and self.direction != 0:  # Down
            self.direction = 2
        elif action == 3 and self.direction != 1:  # Left
            self.direction = 3

        # Calculate new head position
        head_x, head_y = self.snake[0]
        if self.direction == 0:  # Up
            head_x -= 1
        elif self.direction == 1:  # Right
            head_y += 1
        elif self.direction == 2:  # Down
            head_x += 1
        elif self.direction == 3:  # Left
            head_y -= 1

        new_head = (head_x, head_y)

        # Check for collisions
        if (
            head_x < 0 or head_y < 0 or
            head_x >= self.grid_size or head_y >= self.grid_size or
            new_head in self.snake
        ):
            self.done = True
            return self._get_observation(), -1, self.done, {}

        # Update snake position
        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 1
            self.food = self._place_food()  # Place new food
        else:
            self.snake.pop()
            reward = 0

        return self._get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) == self.food:
                    print('F', end=' ')
                elif (row, col) in self.snake:
                    print('S', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()
        