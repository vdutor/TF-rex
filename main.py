from tools import process, plot_progress
import dqn
import game_agent
import numpy as np
import numpy.random as rnd
import time

## Constants
height = 50
width = 100
memory_size = 2000
explore_prob = 1.
explore_stop = .01
explore_anneling = 1000
explore_delta = (explore_prob - explore_stop) / explore_anneling
pretraining_steps =  100
batch_size = 32
training_hz = 5
discount = .99
num_epoch = 1000
len_epoch = 10000
num_actions = len(game_agent.GameAgent.actions)
save_hz = 20 * 60   # save every 20 min
path = "./results1" # relative path, where results are stored


memory = dqn.Memory(memory_size)
network = dqn.DQN(height, width, num_actions, path)
agent = game_agent.GameAgent("127.0.0.1", 9090)

steps_epoch = []
reward_epoch = []
steps_to_train = training_hz
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
        action = rnd.randint(num_actions) if explore else network.get_action(state)
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
            target[~crashes] += discount * network.get_action_and_q(states_next[~crashes])[1]
            network.train(states, actions, target)

            steps_to_train = training_hz

        steps_to_train -= 1
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
