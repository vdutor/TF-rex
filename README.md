# TF-rex
In this project we play Google's T-rex game using Reinforcement Learning.
The RL algorithm is based on the Deep Q-Learning algorithm [1] and is implemented in TensorFlow.

see the accompanying (blogpost)[https://vdutor.github.io/blog/2018/01/05/TF-rex.html]

## Dependencies
 - __Python 3.5 or higher__
 - Pillow 4.3.0
 - scipy 0.19.1
 - tensorflow 1.4.0
 - optional: tensorflow tensorboard


## Installation

Tested on MacOs, Debian, Ubuntu, and Ubuntu-based distros.

Start by cloning the repositery
```sh
$ git clone https://github.com/vdutor/TF-rex
```

We recommend creating a virtualenv before installing the required packages. See [virtualenv](https://virtualenv.pypa.io/en/stable/) or [virtualenv-wrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) on how to do so.

The dependencies can be easly installed using pip.
```sh
$ optional: open the virtualenv
$ pip install -r requirements.txt
```
### Webserver for running the javascript T-rex game

A simple webserver is required to run the T-rex game. We propose two different ways to set this up.
The approaches are tested for the Google Chrome and the Mozilla Firefox browser.

__1. Python's Simple Webserver__

This easiest way to start a local webserver is using python Simple HTTP Server.
Open a new terminal and go inside the t-rex-runner directory to run the following command
```sh
$ cd /path/to/project/TF-Rex/t-rex-runner
$ python2 -m SimpleHTTPServer 8000 
```
The game is now accessable on your localhost `127.0.0.1:8000`.

__2. Alternative: Apache webserver__

Alternatively, an Apache Webserver can be used
```sh
$ sudo apt install apache2
$ cd /var/www
$ sudo rm -r html
$ sudo ln -s /path/to/project/TF-Rex/t-rex-runner html
```
The game is now accessable on your localhost `127.0.0.1`. 

## Getting started

First, all the commandline arguments can be retrieved with
```sh
$ python main.py --help
```

Quickly check if the installation was successful by playing with a pretrained Q-learner.
```sh
$ python main.py --notraining --logdir ./models/model-3-actions
```
This command will restore the pretrained model, stored in `models/model-3-actions` and play the T-rex game.
IMPORTANT: The browser needs to connect with the python program. Therefore, the browserpage, hosting the T-rex game, needs to be refreshed.

Training a new model can be done as follows
```sh
$ python main.py --logdir logs
```
Again, the browser needs to be refreshed to start the process. The directory passed as `logdir` argument, here `logs`, will be used to store intermediate tensorflow checkpoints and tensorboard information.

While training, a different terminal can be opened to launch the tensorboard
```sh
$ tensorboard --logdir logs
```
The tensorboards will be visible on `http://127.0.0.1:6006/`.

## References
[1] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

### TODO
- add Q values as summary histograms in tensorboard
- try without convolutional layers
- write blogpost
