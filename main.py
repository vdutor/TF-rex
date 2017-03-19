from tools import processImage
import dqn
import game_agent
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

## Constants
height = 50
width = 100
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
actions = game_agent.GameAgent.actions
num_actions = len(actions)

memory = dqn.Memory(memory_size)
network = dqn.DQN(height, width, num_actions)
agent = game_agent.GameAgent("127.0.0.1", 9090)
