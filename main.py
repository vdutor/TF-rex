from game_agent import GameAgent, Action
from PIL import Image
import numpy as np
import time

agent = GameAgent('localhost', 9090)
agent.startGame()

for cnt in range(20):
    image, reward, crashed = agent.doAction(Action.UP)
    print crashed
    # Image.fromarray(image).save("figure_test_" + str(cnt) + ".png")
    # time.sleep(.3)
