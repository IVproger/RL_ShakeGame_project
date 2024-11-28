def compute_metrics(agent, env, num_simulations=10):
    # Initialize lists to store metrics
    snake_lengths = []
    snake_lifetimes = []
    episode_rewards = []

    for sim in range(num_simulations):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            print(f"Reward: {reward}")

        # Track metrics
        snake_length = env.get_snake_length()
        snake_lengths.append(snake_length)
        episode_rewards.append(episode_reward)
    
    # return dictionary of metrics
    return {'snake_lengths': snake_lengths, 'episode_rewards': episode_rewards}