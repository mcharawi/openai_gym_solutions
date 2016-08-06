import numpy as np

class Agent(object):

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    def __init__(self, observation, reward, info, action, action_space, done=True):
        self.q_matrix = np.random.rand(16, 4)
        self.gamma = 0.5
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.action = action
        self.action_space = action_space

    def update(self, observation, reward, info, action, action_space, done=True):
        old_observation = self.observation
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.action = action
        self.action_space = action_space
        self.q_matrix[old_observation, action] = reward + self.gamma * np.amax(self.q_matrix[observation, :])
        print reward
        print self.q_matrix


    def take_action(self):
        return np.argmax(self.q_matrix[self.observation, :])


