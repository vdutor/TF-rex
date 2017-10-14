from agent import DDQNAgent
from game_agent import GameAgent
from input_processor import InputProcessor
import numpy as np
import numpy.random as rnd
import tensorflow as tf
import os

## RL Constants
width = 150
height = 50
num_epoch = 1000
len_epoch = 100000
num_actions = len(GameAgent.actions)

## Application flags
tf.app.flags.DEFINE_string("path", "./logs/", "Path to store session checkpoints and tensorboard summaries")
tf.app.flags.DEFINE_integer("checkpoint_hz", 200, "Creating a checkpoint every x epochs")
tf.app.flags.DEFINE_boolean("training", True, "Train a new model")
tf.app.flags.DEFINE_boolean("visualize", True, "Visualize")
tf.app.flags.DEFINE_string("checkpoint_name", "./logs/checkpoints/tf-rex.ckpt-2", "Name of a checkpoint to load")
FLAGS = tf.app.flags.FLAGS

def check_path_existance(path):
    if not path: return
    if os.path.exists(path):
        print("PATH FOR STORING RESULTS ALREADY EXISTS!")
        exit(1)
    os.makedirs(path)

def setup_summary():
    with tf.variable_scope("statistics"):
        summary_scalars = ["exploration", "ep_steps", "ep_reward"]
        summary_placeholders, summary_ops = {}, {}
        for tag in summary_scalars:
            summary_placeholders[tag] = tf.placeholder('float32', None)
            summary_ops[tag]  = tf.summary.scalar(tag, summary_placeholders[tag])
    return summary_ops, summary_placeholders

def summarize(session, writer, cnt, summary_ops, summary_placeholders, values):
    ops = [summary_ops[tag] for tag in list(values.keys())]
    feed_dict = {summary_placeholders[tag]: values[tag] for tag in list(values.keys())}
    summary_lists = session.run(ops, feed_dict)
    for summary in summary_lists:
        writer.add_summary(summary, cnt)

def main(_):
    check_path_existance(FLAGS.path)
    tf.reset_default_graph()
    session = tf.Session()
    writer = tf.summary.FileWriter(FLAGS.path, session.graph) if FLAGS.visualize else None
    summary_ops, summary_placeholders = setup_summary()

    game_agent = GameAgent("127.0.0.1", 9090)
    network = DDQNAgent(session, num_actions, width, height, FLAGS.path, writer)
    processor = InputProcessor(width, height)

    if not FLAGS.training:
        network.load(FLAGS.checkpoint_name)
    network.update_target_network()

    for epoch in range(num_epoch):
        print("\nEpoch: ", epoch)

        state,_,crashed = game_agent.start_game()
        state = processor.process(state)
        ep_steps, ep_reward = 0, 0

        while ep_steps < len_epoch:

            action = network.act(state)
            state_next, reward, crashed = game_agent.do_action(action)
            print("action: {}\t crashed: {}".format(GameAgent.actions[action], crashed))
            state_next = processor.process(state_next)
            network.remember(state, action, reward, state_next, crashed)

            if crashed:
                break

            state = state_next

            ep_steps += 1
            ep_reward += reward

        network.replay(epoch)
        network.explore_less()
        network.update_target_network()

        stats = {"exploration": network.explore_prob, "ep_steps": ep_steps, "ep_reward": ep_reward}
        summarize(session, writer, epoch, summary_ops, summary_placeholders, stats)

        if (epoch+1) % FLAGS.checkpoint_hz == 0:
            network.save(epoch)

if __name__ == '__main__':
    tf.app.run()
