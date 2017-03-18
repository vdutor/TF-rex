import numpy as np
import tensorflow as tf

class DQN:

    def __init__(self, height, width, num_actions):
        self.num_actions = num_actions
        # Input Layer
        self.input =  tf.placeholder(shape=[None, height, width],dtype=tf.float32)

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d( inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = f.contrib.layers.flatten(pool2)
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

        # all Q values
        self.Qs = tf.layers.dense(inputs=dense, units=num_actions)
        # action with highest Q values
        self.a = tf.argmax(Qs, 1)
        # Q value belonging to selected action
        self.Q = tf.reduce_max(Qs, 1)

    def best_action(self):
        return self.a


    def loss(self):
        Q_target = tf.placeholder(shape=[None],dtype=tf.float32)
        actions = tf.placeholder(shape=[None],dtype=tf.int32)
        actions_onehot = tf.one_hot(actions, num_actions, on_value=1, off_value=0, axis=1, dtype=tf.float32)

        # Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)
        # td_error = tf.square(Q_target - self.Q)
        # loss = tf.reduce_mean(self.td_error)
        # trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # updateModel = self.trainer.minimize(self.loss)
