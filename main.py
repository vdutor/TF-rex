from agent import DDQNAgent
from game_agent import GameAgent
from input_processor import InputProcessor
import numpy as np
import numpy.random as rnd
import tensorflow as tf
import os

## RL Constants
width = 80
height = 80
num_epoch = 1000
len_epoch = 1E8
num_actions = len(GameAgent.actions)

## Application flags
tf.app.flags.DEFINE_string("path", "./logs2/", "Path to store the model and tensorboard logs or restore the model.")
tf.app.flags.DEFINE_integer("checkpoint_hz", 100, "Creating a checkpoint every x epochs")
tf.app.flags.DEFINE_boolean("training", True, "Train a new model")
tf.app.flags.DEFINE_boolean("visualize", True, "Visualize")
tf.app.flags.DEFINE_string("checkpoint_name", "./logs2/rex.ckpt", "path of a checkpoint to load")
FLAGS = tf.app.flags.FLAGS

def check_path_existance(path, training):

    if training and os.path.exists(path):
        print("PATH FOR STORING RESULTS ALREADY EXISTS - Results would be overwritten.")
        exit(1)
    elif not training and not os.path.exists(path):
        print("TRAINED MODEL NOT FOUND")
        exit(1)
    elif training and not os.path.exists(path):
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
    check_path_existance(FLAGS.path , FLAGS.training)
    tf.reset_default_graph()
    session = tf.Session()
    writer = tf.summary.FileWriter(FLAGS.path, session.graph) if FLAGS.visualize else None
    summary_ops, summary_placeholders = setup_summary()

    game_agent = GameAgent("127.0.0.1", 9090)
    network = DDQNAgent(session, num_actions, width, height, FLAGS.path, writer)
    processor = InputProcessor(width, height)

    # Playing, assuming we can load a pretrained network
    if not FLAGS.training:
        network.load(FLAGS.checkpoint_name)

        while True:
            ep_steps, ep_reward = network.play(game_agent, processor)

        exit(0)

    # Training...
    network.update_target_network()

    for epoch in range(num_epoch):
        print("\nEpoch: ", epoch)

        frame, _ , crashed = game_agent.start_game()
        frame = processor.process(frame)
        state = np.array([frame, frame, frame, frame])
        ep_steps, ep_reward = 0, 0

        while ep_steps < len_epoch:

            action, explored = network.act(state)
            next_frame, reward, crashed = game_agent.do_action(action)
            print("action: {}\t crashed: {}".format(GameAgent.actions[action] + ["", "*"][explored], crashed))
            next_frame = processor.process(next_frame)
            next_state = np.array([*state[-3:], next_frame])
            network.remember(state, action, reward, next_state, crashed)

            ep_steps += 1
            ep_reward += reward

            if crashed:
                break

            state = next_state

        network.replay(epoch)
        network.explore_less()
        network.update_target_network()

        stats = {"exploration": network.explore_prob, "ep_steps": ep_steps, "ep_reward": ep_reward}
        summarize(session, writer, epoch, summary_ops, summary_placeholders, stats)

        if (epoch+1) % FLAGS.checkpoint_hz == 0:
            network.save(epoch)

if __name__ == '__main__':
    tf.app.run()
