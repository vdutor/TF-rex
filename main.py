from agent import DDQNAgent
from game_agent import GameAgent
from input_processor import InputProcessor
import numpy as np
import numpy.random as rnd

## Constants
width = 100
height = 50
num_epoch = 1000
len_epoch = 100000
num_actions = len(GameAgent.actions)
path = "./results2" # relative path, where results are stored

game_agent = GameAgent("127.0.0.1", 9090)
network = DDQNAgent(num_actions, width, height)
processor = InputProcessor(width, height)

for epoch in range(num_epoch):
    print "\nEpoch: ", epoch

    state,_,crashed = game_agent.start_game()
    state = processor.process(state)
    step, total_reward = 0, 0

    while step < len_epoch:

        action = network.act(state)
        state_next, reward, crashed = game_agent.do_action(action)
        print "action: {}\t crashed: {}".format(GameAgent.actions[action], crashed)
        state_next = processor.process(state_next)
        network.remember(state, action, reward, state_next, crashed)

        if crashed:
            break

        step += 1
        total_reward += reward

    network.replay()
    network.explore_less()
    network.update_target_network()

    print "Number of steps in epoch: ", step
    print "Total reward in epoch :", total_reward
