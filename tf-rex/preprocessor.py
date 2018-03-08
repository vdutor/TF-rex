from scipy.misc import imresize
import numpy as np


class Preprocessor:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def process(self, frame):
        roi_height, roi_width = frame.shape[0], int(frame.shape[1] * .68)
        processed = np.zeros((roi_height, roi_width))

        roi = frame[:, :roi_width, 0]
        all_obstacles_idx = roi > 50
        processed[all_obstacles_idx] = 1
        unharmful_obstacles_idx = roi > 200
        processed[unharmful_obstacles_idx] = 0

        processed = imresize(processed, (self.height, self.width, 1))
        processed = processed / 255.0
        return processed

    def get_initial_state(self, first_frame):
        self.state = np.array([first_frame, first_frame, first_frame, first_frame])
        return self.state

    def get_updated_state(self, next_frame):
        self.state =  np.array([*self.state[-3:], next_frame])
        return self.state
