def processImage(image):
    processed = np.zeros((image.shape[0], image.shape[1]/2))

    roi = image[:,:300,0]
    all_obstacles_idx = roi > 50
    processed[all_obstacles_idx] = 1
    unharmful_obstacles_idx = roi > 200
    processed[unharmful_obstacles_idx] = 0

    processed = imresize(processed, (height, width, 1))
    processed = processed / 255.0
    return processed


