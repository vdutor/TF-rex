import numpy as np
import tensorflow as tf
import random
import os

class DQN:

    def __init__(self, height, width, num_actions, path=None):

        if path is not None and os.path.exists(path):
            print "PATH FOR STORING RESULTS ALREADY EXISTS!"
            exit(1)
        os.makedirs(path)

        self.save_cnt = 0
        self.path = path
        self.num_actions = num_actions
        self.height = height
        self.width = width

        self._create_network()

        # Prepare session
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(init)

    def get_action_and_q(self, states):
        """
        returns array:
            array[0]: actions: is a array of length len(state) with the action with the highest score
            array[1]: q value: is a array of length len(state) with the Q-value belonging to the action
        """
        states = states.reshape(-1, self.height, self.width, 1)
        return self.session.run([self.a, self.Q], {self.state: states})

    def get_action(self, states):
        """
        returns action(s),
            - if states contains only a single state then we return the optimal action as an integer,
            - if states contains an array of states then we return the optimal action for each state of the array
        """
        states = states.reshape(-1, self.height, self.width, 1)
        num_states = states.shape[0]
        actions = self.session.run(self.a, {self.state: states})
        return actions[0] if num_states == 1 else actions

    def train(self, states, actions, targets):
        states = states.reshape(-1, self.height, self.width, 1)
        feed_dict = {self.state: states, self.actions: actions, self.Q_target: targets}
        self.session.run(self.minimize, feed_dict)

    def save(self):
        self.saver.save(self.session, self.path + '/model', global_step = self.save_cnt)
        self.save_cnt += 1

    def _create_network(self):
        tf.reset_default_graph()

        # Input Layer
        self.state =  tf.placeholder(shape=[None, self.height, self.width, 1],dtype=tf.float32)

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=self.state, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d( inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.contrib.layers.flatten(pool2)
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

        # all Q values
        self.Qs = tf.layers.dense(inputs=dense, units=self.num_actions, activation=None)
        # action with highest Q values
        self.a = tf.argmax(self.Qs, 1)
        # Q value belonging to selected action
        self.Q = tf.reduce_max(self.Qs, 1)

        # For training
        self.Q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        actions_onehot = tf.one_hot(self.actions, self.num_actions, on_value=1., off_value=0., axis=1, dtype=tf.float32)

        Q_tmp = tf.reduce_sum(tf.multiply(self.Qs, actions_onehot), axis=1)
        loss = tf.reduce_mean(tf.square(self.Q_target - Q_tmp))
        optimizer = tf.train.AdamOptimizer()
        self.minimize = optimizer.minimize(loss)


class Memory:

    def __init__(self, size):
        self.size = size
        self.mem = np.ndarray((size,5), dtype=object)
        self.iter = 0
        self.current_size = 0

    def add(self, state1, action, reward, state2, crashed):
        self.mem[self.iter,:] = state1, action, reward, state2, crashed
        self.iter = (self.iter + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, n):
        n = min(self.current_size, n)
        random_idx = random.sample(range(self.current_size), n)
        sample = self.mem[random_idx]
        return (np.stack(sample[:,i], axis=0) for i in range(5))
