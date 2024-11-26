import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class SimpleSnakeEnv(gym.Env):
    def __init__(self, grid_size=10, food_reward=75, collision_reward=-75, final_reward=200):
        super(SimpleSnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(0, 1, (11,), dtype=np.float32)
        self.reset()
        self.seed()
        self.food_reward = food_reward
        self.collision_reward = collision_reward
        self.final_reward = final_reward
    
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]  # Initial snake position
        self.free_cells = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size))
        self.free_cells.remove(self.snake[0])
        self.food = self._place_food()
        self.direction = 1  # Initial direction: Right
        self.done = False
        self.previous_distance = self._calculate_distance(self.snake[0], self.food)
        return self._get_observation()

    def _place_food(self):
        food_pos = random.choice(list(self.free_cells))
        return food_pos

    def _get_observation(self):
        # Snake head position
        head_x, head_y = self.snake[0]

        # Direction vectors for movement
        directions = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }

        # Danger detection (collision risk in each direction)
        danger_straight = self._is_danger(head_x + directions[self.direction][0], head_y + directions[self.direction][1])
        danger_right = self._is_danger(
            head_x + directions[(self.direction + 1) % 4][0],
            head_y + directions[(self.direction + 1) % 4][1]
        )
        danger_left = self._is_danger(
            head_x + directions[(self.direction - 1) % 4][0],
            head_y + directions[(self.direction - 1) % 4][1]
        )

        # Current direction (one-hot encoding)
        direction_up = int(self.direction == 0)
        direction_right = int(self.direction == 1)
        direction_down = int(self.direction == 2)
        direction_left = int(self.direction == 3)

        # Food direction
        food_x, food_y = self.food
        food_up = int(food_x < head_x)
        food_down = int(food_x > head_x)
        food_left = int(food_y < head_y)
        food_right = int(food_y > head_y)

        # Concatenate all features into a single state vector
        state = np.array([
            danger_straight, danger_right, danger_left,
            direction_up, direction_right, direction_down, direction_left,
            food_up, food_down, food_left, food_right
        ], dtype=np.float32)

        return state

    def _is_danger(self, x, y):
        """Helper function to check if a position is a danger (wall or snake body)."""
        return (
            x < 0 or y < 0 or
            x >= self.grid_size or y >= self.grid_size or
            (x, y) in self.snake
    )

    def _calculate_distance(self, position, food_position):
        """Calculate Manhattan distance between the snake's head and the food."""
        return abs(position[0] - food_position[0]) + abs(position[1] - food_position[1])

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
            return self._get_observation(), self.collision_reward, self.done, {}

        # Update snake position
        self.snake.insert(0, new_head)
        self.free_cells.remove(new_head)

        if len(self.free_cells) == 0:
            self.done = True
            reward = self.final_reward
            return self._get_observation(), reward, self.done, {}

        # Calculate distance-based reward
        current_distance = self._calculate_distance(new_head, self.food)
        distance_reward = self.previous_distance - current_distance
        # distance_reward = 0

        if new_head == self.food:
            reward = self.food_reward + distance_reward  # Extra reward for eating food
            self.food = self._place_food()  # Place new food
            self.previous_distance = self._calculate_distance(new_head, self.food)
        else:
            tail = self.snake.pop()
            self.free_cells.add(tail)
            reward = distance_reward
            self.previous_distance = current_distance  # Update the previous distance

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
