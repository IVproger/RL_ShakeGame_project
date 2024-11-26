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

    def __init__(self, grid_size=10, food_reward=10, collision_reward=-10, final_reward=100,interact=True):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(0, 1, (grid_size, grid_size, 3), dtype=np.float32)
        self.seed()
        self.reset()
        self.food_reward = food_reward
        self.collision_reward = collision_reward
        self.final_reward = final_reward

        # Pygame initialization
        if interact:
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
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for x, y in self.snake:
            obs[x, y] = [0, 1, 0]  # Snake body in green
        obs[self.food[0], self.food[1]] = [1, 0, 0]  # Food in red
        return obs

    def _calculate_distance(self, position, food_position):
        """Calculate Manhattan distance between the snake's head and the food."""
        return abs(position[0] - food_position[0]) + abs(position[1] - food_position[1])

    def _change_direction(self, action):
        if action == self.UP:
            self.direction = (-1, 0)  # Move up
        elif action == self.RIGHT:
            self.direction = (0, 1)   # Move right
        elif action == self.DOWN:
            self.direction = (1, 0)   # Move down
        elif action == self.LEFT:
            self.direction = (0, -1)  # Move left

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called after environment is done")

        # Update direction based on action
        self._change_direction(action)

        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collisions
        if (
            new_head[0] < 0 or new_head[1] < 0 or
            new_head[0] >= self.grid_size or new_head[1] >= self.grid_size or
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
        pygame.quit()

def human_mode():
    env = SnakeEnv(grid_size=10)
    env.reset()
    
    action = env.RIGHT  # Start with the RIGHT action

    while True:  # Run until the user quits
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = env.UP
                elif event.key == pygame.K_RIGHT:
                    action = env.RIGHT
                elif event.key == pygame.K_DOWN:
                    action = env.DOWN
                elif event.key == pygame.K_LEFT:
                    action = env.LEFT
                elif event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    env.close()
                    return
        
        observation, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.5) 
        
        if done:
            print("Game over!")
            env.reset()


if __name__ == "__main__":
    human_mode()