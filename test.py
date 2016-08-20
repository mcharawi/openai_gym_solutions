__author__ = 'tushar'
import gym
from agent import Agent
import json

with open('API.json') as api:
    data = json.load(api)

env = gym.make('FrozenLake-v0')
env.monitor.start('/tmp/frozenlake-experiment-9')

observation = env.reset()
action_space = env.action_space
observation, reward, done, info = env.step(action_space.sample())
agent = Agent(observation, reward, info, action_space.sample(), action_space)
num_episodes = 5000

for i_episode in range(num_episodes):
    observation = env.reset()
    done = False
    t = 0
    while not done:
        env.render()
        action = agent.take_action()
        (observation, reward, done, info) = env.step(action)
        agent.update(observation, reward, info, action, action_space)
        t = t + 1
        if done:
            break
            print("Episode finished after {} timesteps".format(t+1))

env.monitor.close()
gym.scoreboard.api_key = data["api_key"]
gym.upload('/tmp/frozenlake-experiment-9')