import numpy as np
import random

class Agent(object):

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    def __init__(self, observation, reward, info, action, action_space, done=True):
        self.q_matrix = np.random.rand(16, 4)
        self.alpha = 0.5
        self.gamma = 0.5
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.action = action
        self.action_space = action_space
        self.epsilon = 1
        self.decay = 0.98

    def update(self, observation, reward, info, action, action_space, done=True):
        old_observation = self.observation

        self.observation = observation
        R = 0
        if reward == 0 and done:
            R = -100
        elif reward != 0 and done:
            R = 100
        self.reward = R
        self.done = done
        self.info = info
        self.action = action
        self.action_space = action_space
        self.q_matrix[old_observation, action] = self.q_matrix[old_observation, action] + \
                                                self.alpha * (self.reward + self.gamma * np.amax(self.q_matrix[observation, :]) -
                                                self.q_matrix[old_observation, action])

        #print self.q_matrix
        self.epsilon *= self.decay

    def take_action(self):
        if random.random() > self.epsilon:
            return np.argmax(self.q_matrix[self.observation, :])
        else:
            return self.action_space.sample()
        # return np.argmax(self.q_matrix[self.observation, :])


