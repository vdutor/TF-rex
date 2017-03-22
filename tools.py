from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def process(image, height, width):
    roi_height, roi_width = image.shape[0], int(image.shape[1] * .6)
    print roi_width
    processed = np.zeros((roi_height, roi_width))

    roi = image[:,:roi_width,0]
    all_obstacles_idx = roi > 50
    processed[all_obstacles_idx] = 1
    unharmful_obstacles_idx = roi > 200
    processed[unharmful_obstacles_idx] = 0

    processed = imresize(processed, (height, width, 1))
    processed = processed / 255.0
    return processed


def plot_progress(epoch_rewards, epoch_length, path):
    num_epochs = len(epoch_rewards)
    x = range(num_epochs)

    # total rewards in function of the epoch number
    plt.plot(x, epoch_rewards, c='red', label='reward')
    plt.savefig(path + "/reward_" + str(num_epochs) + ".png", format = "png")
    plt.clf()

    # total number of steps in function of the epoch number
    plt.plot(x, epoch_length, c='red', label='reward')
    plt.savefig(path + "/length_" + str(num_epochs) + ".png", format = "png")
    plt.clf()


def conv2d(x, output_dim, kernel_shape, stride, name):
    stride = [1, stride[0], stride[1], 1]

    with tf.variable_scope(name):
        w = tf.Variable(tf.truncated_normal(kernel_shape, 0, .02), dtype=tf.float32, name="w")
        conv = tf.nn.conv2d(x, w, stride, "VALID")
        b = tf.Variable(tf.zeros([output_dim]), name="b")
        out = tf.nn.bias_add(conv, b)
        out = tf.nn.relu(out)

    return out, w, b

def linear(x, output_size, name, activation_fn=tf.nn.relu):
    shape = x.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.Variable(tf.random_normal([shape[1], output_size], stddev=.02), dtype=tf.float32, name='w')
        b = tf.Variable(tf.zeros([output_size]), name='b')
        out = tf.nn.bias_add(tf.matmul(x, w), b)

        if activation_fn != None:
            out =  activation_fn(out)

    return out, w, b
