__author__ = 'tushar'
import gym
from agent import Agent

env = gym.make('FrozenLake-v0')


observation = env.reset()
action_space = env.action_space
observation, reward, done, info = env.step(action_space.sample())
agent = Agent(observation, reward, done, info, action_space)

for i_episode in range(20):
    observation = env.reset()
    done = False
    t = 0
    while not done:
        env.render()
        action = agent.action()
        (observation, reward, done, info) = env.step(action)
        agent.update(observation, reward, done, info, action_space)
        t = t + 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break