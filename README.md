---

# Snake Game with Reinforcement Learning

This project was developed as part of the **Reinforcement Learning course at Innopolis University (2024)**. It focuses on applying different RL algorithms to emulate and solve the classic Snake game. The algorithms implemented include Q-learning, SARSA, Deep Q-Network (DQN), and Policy Gradient (REINFORCE). The goal is to optimize the agent (snake) to maximize its score while experimenting with various reward functions and comparing the effectiveness of each algorithm.

## Project Description
The Snake game is a grid-based environment where the agent (snake) moves to eat food while avoiding collisions with itself or the boundaries. The goal is to use RL techniques to teach the agent to maximize its score by strategically selecting actions (up, down, left, right). The project involves simulating the game environment and applying multiple RL algorithms to find the optimal solution for the snake's behavior.

## Contributors
- **Ivan Golov**  
- **Roman Makeev**  
- **Maxim Martyshov**

## Environment Description
The Snake game environment has been simulated using `Pygame`. The **state space** is represented by the snake’s body position, the location of the food, and additional features such as the distance to walls or potential collisions. The **action space** consists of four discrete movements: up, down, left, and right.

Various reward functions have been explored, such as:
- Positive reward for eating food
- Negative reward for collisions
- Distance-based rewards for approaching food

## RL Algorithms
The following Reinforcement Learning algorithms have been implemented and compared in this project:
1. **Q-learning**: A value-based method for model-free control.
2. **SARSA**: On-policy version of Q-learning.
3. **Deep Q-Network (DQN)**: A neural network-based approach to approximate Q-values.
4. **Policy Gradient (REINFORCE)**: A policy-based method to directly learn the optimal policy.

Each algorithm has been tested with various hyperparameters, and their performance has been compared based on the agent’s ability to maximize the score in the Snake game.

## Project Setup
To set up and run the project locally, follow these steps:

> **Note**: This project requires Python version **3.11 or higher**.

1. **Clone the repository**:
   ```bash
   git clone git@github.com:IVproger/RL_ShakeGame_project.git
   cd RL_ShakeGame_project
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -Ur requirements.txt
   ```

## References
- [Snake Game Genetic Algorithm with TensorFlow by Mauro Comi](https://github.com/maurock/snake-ga-tf/tree/master)
- [Deep Reinforcement Learning Algorithms by Rafael Sampaio](https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master)

## Acknowledgments
We would like to thank the following open-source projects for their inspiration and valuable resources:
- **[Mauro Comi's Snake Game Genetic Algorithm with TensorFlow](https://github.com/maurock/snake-ga-tf)**: This project helped us understand the structure and strategy of applying algorithms to the Snake game.
- **[Rafael Sampaio's Deep Reinforcement Learning Algorithms](https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms)**: This repository provided insights into the implementation of RL algorithms in Python.

---
