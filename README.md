Here's a basic structure for your `README.md` file:

---

# Snake Game with Reinforcement Learning

This project was developed as part of the **Reinforcement Learning course at Innopolis University (2024)**. It focuses on applying different RL algorithms to emulate and solve the classic Snake game. The algorithms implemented include Q-learning, SARSA, Deep Q-Network (DQN), and Policy Gradient (REINFORCE). The aim is to optimize the agent (snake) to maximize its score while experimenting with various reward functions and comparing the effectiveness of each algorithm.

## Table of Contents
- [Project Description](#project-description)
- [Algorithms Implemented](#algorithms-implemented)
- [Environment](#environment)
- [Contributors](#contributors)
- [Project Setup](#project-setup)

## Project Description
The Snake game is a grid-based environment where the agent (snake) moves to eat food while avoiding collisions with itself or the boundaries. The goal is to use RL techniques to teach the agent to maximize its score by strategically selecting actions (up, down, left, right). This project evaluates multiple RL algorithms and compares their performance under different reward structures.

## Algorithms Implemented
The following RL algorithms have been implemented in this project:
1. **Q-learning**
2. **SARSA**
3. **Deep Q-Network (DQN)**
4. **Policy Gradient (REINFORCE)**

## Environment
The Snake game environment has been simulated using `Pygame`, with the state space representing the snake's body position and food location, and the action space consisting of four possible movements (up, down, left, right). Various reward functions, such as distance-based and time-based rewards, have been explored.

## Contributors
- **Ivan Golov**  
- **Roman Makeev**  
- **Maxim Martyshov**

## Project Setup
To set up and run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/snake-rl.git
   cd snake-rl
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
---

