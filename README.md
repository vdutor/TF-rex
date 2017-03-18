# TF-rex
In this project we try to play Google's T-rex game using Reinforcement Learning. The RL algorithm is based on the Deep Q-Learning algorithm [1] and will be implemented in TensorFlow.

### Dependencies
 - Pillow 4.0.0
 - PyUserInput 0.1.11
 - scipy 0.18.1
 - tensorflow 1.0.0
 
### Setup

For Debian, Ubuntu, and Ubuntu-based distro:
Install necessary packages to have a simple web server and configure the server to use the right pages
```sh
$ sudo apt-get install apache2
$ cd /var/www
$ sudo ln -s /path/to/project/TF-Rex/t-rex-runner html
```
The game is now accessable on `127.0.0.1`

Next, install the Chrome plugin
[Allow-Control-Allow-Origin: *](https://chrome.google.com/webstore/detail/allow-control-allow-origi/nlfbmbojpeacfghkpbjhddihlkkiljbi?hl=en)

### References
[1] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
