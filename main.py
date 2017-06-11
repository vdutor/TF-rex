from tools import process, plot_progress
from agent import DDQNAgent
from game_agent import GameAgent
from input_processor import InputProcessor
import numpy as np
import numpy.random as rnd
import time
import tensorflow as tf

import matplotlib.pyplot as plt

## Constants
width = 100
height = 50
num_epoch = 1000
len_epoch = 100000
num_actions = len(GameAgent.actions)
path = "./results2" # relative path, where results are stored


game_agent = GameAgent("127.0.0.1", 9090)
agent = DDQNAgent(num_actions, width, height)
processor = InputProcessor(width, height)


steps_epoch = []
reward_epoch = []
time_last_save = time.time()

for epoch in range(num_epoch):
    print "\nEpoch: ", epoch

    state,_,crashed = agent.start_game()
    state = processor.process(state)
    step = 0
    total_reward = 0

    while step < len_epoch:

        action = agent.act(state)
        state_next, reward, crashed = agent.do_action(action)
        print "action: {}\t crashed: {}".format(game_agent.GameAgent.actions[action], crashed)
        state_next = processor.process(state_next)
        agent.remember(state, action, reward, state_next, crashed)

        if crashed:
            break

        step += 1
        total_reward += reward

    agent.replay()
    agent.explore_less()
    agent.update_target_network()


    steps_epoch.append(step)
    reward_epoch.append(total_reward)
    print "Number of steps in epoch: ", step
    print "Total reward in epoch :", total_reward

    if time.time() - time_last_save > save_hz:
        network.save()
        time_last_save = time.time()
        print "Network saved"

network.save()
