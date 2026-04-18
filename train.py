import gymnasium as gym
import config
from agent import DQNAgent
from replay_buffer import ReplayBuffer


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, replay_buffer)

    scores = []

    for episode in range(config.NUM_EPISODES):
        state, info = env.reset()
        
        total_reward = 0

        while True:
            action = agent.action_selection(state)

            next_state, reward, terminated, truncated ,info = env.step(action)

            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)

            agent.learn()

            state = next_state

            total_reward += reward

            if done:
                break

        
        agent.update_epsilon()

        if(episode % config.TARGET_UPDATE_FREQ == 0):
            agent.update_target_network()

        scores.append(total_reward)

        print(f"Episode no. : {episode + 1} Reward : {total_reward}")

        if len(scores) >= 100:
            avg = sum(scores[-100:]) / 100
            if avg >= config.SOLVED_THRESHOLD:
                print(f"solved in {episode + 1} episodes")
                break

