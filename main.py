from tools import process, plot_progress
import dqn
import game_agent
import numpy as np
import numpy.random as rnd
import time
import tensorflow as tf

import matplotlib.pyplot as plt

## Constants
height = 50
width = 100
memory_size = 100000
explore_prob = 1.
explore_stop = .01
explore_anneling = 1000000
explore_delta = (explore_prob - explore_stop) / explore_anneling
pretraining_steps =  100
batch_size = 32
training_hz = 5
transfer_hz = 1000
discount = .99
num_epoch = 100000
len_epoch = 10000000
num_actions = len(game_agent.GameAgent.actions)
save_hz = 20 * 60   # save every 20 min
path = "./results2" # relative path, where results are stored


tf.reset_default_graph()
session = tf.Session()
agent = game_agent.GameAgent("127.0.0.1", 9090)
memory = dqn.Memory(memory_size)
main_dqn = dqn.DQN(session, height, width, num_actions, "main", None)
target_dqn = dqn.DQN(session, height, width, num_actions, "target", None)

session.run(tf.global_variables_initializer())
target_dqn.tranfer_variables_from(main_dqn)

steps_epoch = []
reward_epoch = []
steps_to_train = training_hz
steps_to_transfer = transfer_hz
time_last_save = time.time()

for epoch in range(num_epoch):
    print "\nEpoch: ", epoch

    state,_,crashed = agent.start_game()
    state = process(state, height, width)
    step = 0
    total_reward = 0

    while step < len_epoch and not crashed:
        # keep track how many pretraining steps we still have to execute
        pretraining_steps = max(pretraining_steps - 1, 0)

        explore = rnd.rand() <= explore_prob or pretraining_steps > 0
        action = rnd.randint(num_actions) if explore else main_dqn.get_action(state)
        state_next, reward, crashed = agent.do_action(action)
        print "action: {}\t crashed: {}\t random action: {} (prob: {})" \
                .format(game_agent.GameAgent.actions[action], crashed, explore, round(explore_prob,2))
        state_next = process(state_next, height, width)

        memory.add(state, action, reward, state_next, crashed)

        if pretraining_steps <= 0 and steps_to_train <= 0:
            print "...\nTraining network\n..."

            # reduce exploration probability, can not be smaller than explore_stop
            explore_prob = max(explore_stop, explore_prob - explore_delta)

            states, actions, rewards, states_next, crashes = memory.sample(batch_size)
            target = rewards
            # add Q value of next state to not terminal states (i.e. not crashed)
            target[~crashes] += discount * target_dqn.get_action_and_q(states_next[~crashes])[1]
            main_dqn.train(states, actions, target)

            steps_to_train = training_hz

        if pretraining_steps <= 0 and steps_to_transfer <= 0:
            print "...\nTransfering networks\n..."
            steps_to_transfer = training_hz
            target_dqn.tranfer_variables_from(main_dqn)


        steps_to_train -= 1
        steps_to_transfer -= 1
        state = state_next
        step += 1
        total_reward += reward

    steps_epoch.append(step)
    reward_epoch.append(total_reward)
    print "Number of steps in epoch: ", step
    print "Total reward in epoch :", total_reward

    if time.time() - time_last_save > save_hz:
        network.save()
        time_last_save = time.time()
        print "Network saved"

        plot_progress(reward_epoch, steps_epoch, path)

network.save()
