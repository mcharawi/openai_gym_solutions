__author__ = 'tushar'

class Agent(object):

    def __init__(self, observation, reward, done, info, action_space):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.action_space = action_space

    def update(self, observation, reward, done, info, action_space):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.action_space = action_space

    def action(self):
        return self.action_space.sample()
