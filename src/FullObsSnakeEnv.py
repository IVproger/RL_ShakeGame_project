import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pygame

class FullObsSnakeEnv(gym.Env):
    
    # Define the actions
    UP = 0    #!< Move up
    RIGHT = 1 #!< Move right
    DOWN = 2  #!< Move down
    LEFT = 3  #!< Move left
    
    def __init__(self, grid_size=10, food_reward=75, collision_reward=-75, final_reward=200, interact=True):
        super(FullObsSnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(0, 1, (grid_size, grid_size, 2), dtype=np.float32)
        self.reset()
        self.seed()
        self.interact = interact
        self.food_reward = food_reward
        self.collision_reward = collision_reward
        self.final_reward = final_reward
        
        # Pygame initialization
        if self.interact:
            pygame.init()
            self.screen_size = 500
            self.cell_size = self.screen_size // self.grid_size
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Snake Environment')
            self.clock = pygame.time.Clock()
    
    def seed(self, seed=None):
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
        food_pos = random.choice(list(self.free_cells))
        return food_pos

    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        for x, y in self.snake:
            obs[x, y] = [0, 1] 
        obs[self.food[0], self.food[1]] = [1, 0] 
        return obs

    def _calculate_distance(self, position, food_position):
        """Calculate Manhattan distance between the snake's head and the food."""
        return abs(position[0] - food_position[0]) + abs(position[1] - food_position[1])

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
        self._update_direction(action)

        # Calculate new head position
        new_head = self._get_new_head_position()

        # Check for collisions
        if self._check_collision(new_head):
            self.done = True
            return self._get_observation(), self.collision_reward, self.done, {}

        # Update snake position
        self._update_snake_position(new_head)

        if len(self.free_cells) == 0:
            self.done = True
            return self._get_observation(), self.final_reward, self.done, {}

        # Calculate distance-based reward
        reward = self._calculate_reward(new_head)

        return self._get_observation(), reward, self.done, {}

    def _update_direction(self, action):
        '''
        Update the direction of the snake based on the action.
        Params
        action : int
            Action to change the direction.
        '''
        if action == 0 and self.direction != 2:  # Up
            self.direction = 0
        elif action == 1 and self.direction != 3:  # Right
            self.direction = 1
        elif action == 2 and self.direction != 0:  # Down
            self.direction = 2
        elif action == 3 and self.direction != 1:  # Left
            self.direction = 3

    def _get_new_head_position(self):
        '''
        Calculate the new head position of the snake.
        Returns
        new_head : tuple
            New head position of the snake.
        '''
        head_x, head_y = self.snake[0]
        if self.direction == 0:  # Up
            head_x -= 1
        elif self.direction == 1:  # Right
            head_y += 1
        elif self.direction == 2:  # Down
            head_x += 1
        elif self.direction == 3:  # Left
            head_y -= 1
        return (head_x, head_y)

    def _check_collision(self, new_head):
        '''
        Check if the new head position results in a collision.
        Params
        new_head : tuple
            New head position of the snake.
        Returns
        bool
            True if there is a collision, False otherwise.
        '''
        head_x, head_y = new_head
        return head_x < 0 or head_y < 0 or head_x >= self.grid_size or head_y >= self.grid_size or new_head in self.snake
    
    def _update_snake_position(self, new_head):
        '''
        Update the snake's position.
        Params
        new_head : tuple
            New head position of the snake.
        '''
        self.snake.insert(0, new_head)
        self.free_cells.remove(new_head)

    def _calculate_reward(self, new_head):
        '''
        Calculate the reward based on the new head position.
        Params
        new_head : tuple
            New head position of the snake.
        Returns
        reward : int
            Reward received after taking the step.
        '''
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

        return reward

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