from game_agent import GameAgent, Action
import dqn
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.misc import imresize
import tensorflow as tf
import random

## Constants
height = 50
width = 100
num_actions = len(GameAgent.actions)
memory_size = 10
explore_start = 1
explore_stop = .01
explore_delta = .05
explore_anneling = 1000
pretraining_steps = 10000
experience_size = 32
training_hz = 10
discount = .99
num_epoch = 100
len_epoch = 10000

class Memory:

    def __init__(self, size):
        self.size = size
        # self.mem = np.array(([None] * 5) * size)
        self.mem = np.ndarray((size,5), dtype=object)
        self.iter = 0

    def add(self, state1, action, reward, state2, crashed):
        self.mem[self.iter,:] = state1, action, reward, state2, crashed
        self.iter = (self.iter + 1) % self.size

    def sample(self, n):
        random_idx = random.sample(range(self.size), n)
        sample = self.mem[random_idx]
        return (np.stack(sample[:,i], axis=0) for i in range(5))


def processImage(image):
    processed = np.zeros((image.shape[0], image.shape[1]/2))

    roi = image[:,:300,0]
    all_obstacles_idx = roi > 50
    processed[all_obstacles_idx] = 1
    unharmful_obstacles_idx = roi > 200
    processed[unharmful_obstacles_idx] = 0

    processed = imresize(processed, (height, width, 1))
    processed = processed / 255.0
    return processed


q_network = dqn.DQN(height, width, num_actions)
agent = GameAgent("127.0.0.1", 9090)

