from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
from io import BytesIO
import SocketServer
import base64
import numpy as np
import multiprocessing
import threading
import time
import pymouse
import pykeyboard
import re

key_up = pykeyboard.PyKeyboard().up_key
key_down = pykeyboard.PyKeyboard().down_key
key_get_state = 'P'

queue = multiprocessing.Queue()


class Action:
    UP = 0
    DOWN = 1
    FORWARD = 2


class GameAgent:
    """
    GameAgent class is responsible for passing the actions to the game.
    For this it uses the pyuserinput module.
    A action is performed in the game by emulating a keypress.
    Besides this the GameAgent class is also responsible for retrieving the game status.
    The logic for this is mostly implemented in the ..Handler.. class.
    """

    def __init__(self, host, port):
        # used to pass the actions
        self.mouse = pymouse.PyMouse()
        self.keyboard = pykeyboard.PyKeyboard()

        # used to retrieve the state of the game
        address = (host, port)
        self.server = HTTPServer(address, Handler)
        print "listening..."
        thread = threading.Thread(target = self.server.serve_forever)
        thread.daemon = True
        thread.start()

    def startGame(self):
        """
        Starts the game and lets the TRex run for half a second and then returns the initial state.

        :return: the initial state of the game (np.array, reward, crashed).
        """
        x_dim, y_dim = self.mouse.screen_size()
        self.mouse.click(x_dim * .9, y_dim/4, 1)
        time.sleep(.5)
        self.keyboard.tap_key(key_up)
        time.sleep(.5)

        return self._get_state()

    def doAction(self, action):
        """
        Performs action and returns the updated status

        :param action:  Must come from the class Action.
                        The only allowed actions are Action.UP, Action.Down and Action.FORWARD.
        :return: return the image of the game after performing the action, the reward (after the action) and
                        whether the TRex crashed or not.
        """
        global queue

        if action == Action.UP:
            self.keyboard.tap_key(key_up)
            time.sleep(.5)
        elif action == Action.DOWN:
            self.keyboard.press_key(key_down)
            time.sleep(.5)
            self.keyboard.release_key(key_down)
        elif action == Action.FORWARD:
            time.sleep(.5)
            pass
        else:
            print "WARNING: wrong action passed to the game agent."
            print "action not executed"

        return self._get_state()

    def _get_state(self):
        self.keyboard.tap_key(key_get_state)

        image, crashed = queue.get()
        reward = -1 if crashed else 1
        return image, reward, crashed


class Handler(BaseHTTPRequestHandler):
    """
    Class for receiving the image (game status) of the game.
    The class forms the Connection between the game's javascript code and the python code.
    The image of the game is sent through a HTTP Post from the js code to the python code.
    This class parses the POST that contains plain texts (url encoded) to python objects
    """

    def __init__(self, request, client_address, server):
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)

    def _set_headers(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        global queue
        self._set_headers()
        content_length = int(self.headers['Content-Length'])
        data = str(self.rfile.read(content_length))

        image, crashed = self._process_post_content(data)
        # place image and death-status in queue
        queue.put((image, crashed))


    def _process_post_content(self, data):
        """
        Ugly parsing...
        TODO: look for other options
        """
        image = re.search('world=(.*)&', data).group(1)

        # remove trailing 'data:image/png;base64,'
        image = re.sub('data%3Aimage%2Fpng%3Bbase64%2C', '', image)
        # from URL encoding to proper base 64
        image = re.sub('%2B', '+', image)
        image = re.sub('%3D', '=', image)
        image = re.sub('%2F', '/', image)
        crashed = re.search('&crashed=(.*)', data).group(1)
        crashed = True if crashed in ['True', 'true'] else False

        # convert image from base64 decoding to np array
        image = np.array(Image.open(BytesIO(base64.b64decode(image))))

        return image, crashed

    def log_message(self, format, *args):
        pass
