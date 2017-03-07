from game_agent import GameAgent, Action
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.misc import imresize
import tensorflow as tf

def processImage(image):
    processed = np.zeros((image.shape[0], image.shape[1]/2))

    roi = image[:,:300,0]
    all_obstacles_idx = roi > 50
    processed[all_obstacles_idx] = 1
    unharmful_obstacles_idx = roi > 200
    processed[unharmful_obstacles_idx] = 0

    processed = imresize(processed, (50, 100))
    processed = processed / 255.0
    processed = np.reshape(processed, (1, -1))
    return processed

agent = GameAgent('localhost', 9090)

# constants
#############
actions = {Action.UP:'up', Action.DOWN:'down', Action.FORWARD:'forward'}

# the model
#############

input_length = 50 * 100

tf.reset_default_graph()
#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,input_length],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_length,len(actions)],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,len(actions)],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        print "Episode: ", i

        #Reset environment and get first new observation
        im,_,_ = agent.startGame()
        s = processImage(im)

        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s})
            if np.random.rand(1) < e:
                a[0] = np.random.randint(len(actions))

            print "Q values in move {}       : {}".format(j,allQ)
            print "Action selected in move {}: {}".format(j,actions[a[0]])
            #Get new state and reward from environment
            im1,r,d = agent.doAction(a[0])
            s1 = processImage(im1)
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s1})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:s,nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                print "Game over"
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"
