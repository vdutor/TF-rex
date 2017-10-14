from scipy.misc import imresize
import numpy as np

class InputProcessor:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def process(self, image):
        roi_height, roi_width = image.shape[0], int(image.shape[1] * .68)
        processed = np.zeros((roi_height, roi_width))

        roi = image[:,:roi_width,0]
        all_obstacles_idx = roi > 50
        processed[all_obstacles_idx] = 1
        unharmful_obstacles_idx = roi > 200
        processed[unharmful_obstacles_idx] = 0

        processed = imresize(processed, (self.height, self.width, 1))
        processed = processed / 255.0
        return processed
