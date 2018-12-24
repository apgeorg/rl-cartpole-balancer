import gym
import numpy as np
from agents.dqn import DQN

def play(gym_id, episodes=1, agent=None):
    env = gym.make(gym_id)
    for e in range(episodes):
        state = env.reset()
        total_reward = 0.
        for t in range(500):
            if agent is None:
                action = env.action_space.sample() # take a random action
            else:
                action = agent.act(np.reshape(state, [1, agent.state_size]))
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                print('Episode {}/{} done in {} steps, total reward {}: '.format(e+1, episodes, t+1, total_reward))
                break
    env.close()

def learn(gym_id, episodes=1000, batch_size=32, model_path="models/model.h5"):
    env = gym.make(gym_id)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = DQN(num_states, num_actions)
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, num_states])
        total_reward = 0.
        for steps in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, num_states])
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                print('Episode {}/{} done in {} steps, total reward {}: '.format(e+1, episodes, steps+1, total_reward))
                if total_reward >= 200:
                    agent.save(model_path)
                    return agent
                break
            if len(agent.memory) > batch_size:
                agent.train(batch_size) # train the agent with the experience of the episode
    env.close()
    return None

if __name__ == '__main__':
    agent = learn('CartPole-v0', episodes=1000, batch_size=24, model_path="./models/cartpole.h5")
    play('CartPole-v0', episodes=5, agent=agent)