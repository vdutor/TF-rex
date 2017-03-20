import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np

def process(image, height, width):
    processed = np.zeros((image.shape[0], image.shape[1]/2))

    roi = image[:,:300,0]
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

