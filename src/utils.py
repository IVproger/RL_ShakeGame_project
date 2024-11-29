from tqdm import tqdm
import json

def compute_metrics(agent, env, save_to, num_simulations = 10) -> dict:
    snake_lengths = []
    episode_rewards = []

    for _ in tqdm(range(num_simulations)):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if env.interact:
                env.render()
               
        # Track metrics
        snake_length = env.get_snake_length()
        snake_lengths.append(snake_length)
        episode_rewards.append(episode_reward)
        print(f"Snake length: {snake_length}, Episode reward: {episode_reward}")
    
    # Create dictionary of metrics
    results = {'snake_lengths': snake_lengths, 'episode_rewards': episode_rewards}
    
    # Save results to JSON file
    with open(save_to, 'w') as f:
        json.dump(results, f)

    return results