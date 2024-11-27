import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pygame
import time

class SnakeEnv(gym.Env):

    # Define the actions
    UP = 0    #!< Move up
    RIGHT = 1 #!< Move right
    DOWN = 2  #!< Move down
    LEFT = 3  #!< Move left

    def __init__(self, grid_size=10, food_reward=75, collision_reward=-75, final_reward=200, interact=True):
        '''
        Initialize the Snake environment.
        Params
        grid_size : int
            Size of the grid.
        food_reward : int
            Reward for eating food.
        collision_reward : int
            Penalty for colliding with walls or itself.
        final_reward : int
            Reward for filling the grid.
        interact : bool
            Whether to enable interactive mode with Pygame.
        '''
        super(SnakeEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(0, 1, (11,), dtype=np.float32)
        self.seed()
        self.reset()
        self.interact = interact
        self.food_reward = food_reward
        self.collision_reward = collision_reward
        self.final_reward = final_reward
        self.direction = None

        # Pygame initialization
        if self.interact:
            pygame.init()
            self.screen_size = 500
            self.cell_size = self.screen_size // self.grid_size
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Snake Environment')
            self.clock = pygame.time.Clock()

    def seed(self, seed=None):
        '''
        Seed the environment's random number generator.
        Params
        seed : int
            Seed for the random number generator.
        '''
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        '''
        Reset the environment to its initial state.
        Returns
        observation : np.array
            Initial observation of the environment.
        '''
        # Randomly initialize the snake's position
        initial_position = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        self.snake = [initial_position]  # Initial snake position
        
        # Initialize the set of free cells and remove the snake's initial position
        self.free_cells = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size))
        self.free_cells.remove(self.snake[0])
        
        self.food = self._place_food()
        self.direction = random.choice([self.UP, self.RIGHT, self.DOWN, self.LEFT]) 
        self.done = False
        self.previous_distance = self._calculate_distance(self.snake[0], self.food)
        return self._get_observation()

    def _place_food(self):
        '''
        Place the food in a random free cell.
        Returns
        food_pos : tuple
            Position of the food.
        '''
        food_pos = random.choice(list(self.free_cells))
        return food_pos

    def _get_observation(self):
        '''
        Get the current observation of the environment.
        Returns
        state : np.array
            Current state of the environment.
        '''
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
        '''
        Helper function to check if a position is a danger (wall or snake body).
        Params
        x : int
            X-coordinate of the position.
        y : int
            Y-coordinate of the position.
        Returns
        bool
            True if the position is a danger, False otherwise.
        '''
        return (
            x < 0 or y < 0 or
            x >= self.grid_size or y >= self.grid_size or
            (x, y) in self.snake
        )

    def _calculate_distance(self, position, food_position):
        '''
        Calculate Manhattan distance between the snake's head and the food.
        Params
        position : tuple
            Position of the snake's head.
        food_position : tuple
            Position of the food.
        Returns
        int
            Manhattan distance between the snake's head and the food.
        '''
        return abs(position[0] - food_position[0]) + abs(position[1] - food_position[1])

    def _change_direction(self, action):
        '''
        Change the direction of the snake based on the action.
        Params
        action : int
            Action to change the direction.
        '''
        if action == self.UP:
            self.direction = (-1, 0)  # Move up
        elif action == self.RIGHT:
            self.direction = (0, 1)   # Move right
        elif action == self.DOWN:
            self.direction = (1, 0)   # Move down
        elif action == self.LEFT:
            self.direction = (0, -1)  # Move left

    def step(self, action):
        '''
        Take a step in the environment based on the action.
        Params
        action : int
            Action to take.
        Returns
        observation : np.array
            Observation after taking the step.
        reward : int
            Reward received after taking the step.
        done : bool
            Whether the episode is done.
        info : dict
            Additional information.
        '''
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
        '''
        Render the environment.
        Params
        mode : str
            Mode of rendering.
        '''
        self.screen.fill((0, 0, 0))  # Fill screen with black
        
         # Draw grid lines
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (self.screen_size, y))

        # Draw food
        food_rect = pygame.Rect(self.food[1] * self.cell_size, self.food[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (220, 0, 0), food_rect)  # Red color for food

        # Draw snake
        for segment in self.snake:
            segment_rect = pygame.Rect(segment[1] * self.cell_size, segment[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 220, 0), segment_rect)  # Green color for snake

        pygame.display.flip()
        self.clock.tick(10)  # Control the frame rate

    def close(self):
        '''
        Close the environment.
        '''
        if self.interact:
            pygame.display.quit()
            pygame.quit()

    def get_snake_length(self):
        '''
        Get the current length of the snake.
        Returns
        int
            Length of the snake.
        '''
        return len(self.snake)