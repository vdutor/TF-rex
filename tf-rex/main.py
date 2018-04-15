from agent import DDQNAgent
from environment import Environment
from preprocessor import Preprocessor
from functools import partial
import numpy as np
import numpy.random as rnd
import tensorflow as tf
import os

## Constants
width = 80
height = 80
len_epoch = int(1E8)
num_actions = len(Environment.actions)

## Application flags
tf.app.flags.DEFINE_string("logdir", "./logs/", "Path to store the model and tensorboard logs or restore the model")
tf.app.flags.DEFINE_string("checkpoint_nr", None, "Checkpoint number of the model to restore")
tf.app.flags.DEFINE_integer("checkpoint_hz", 200, "Creating a checkpoint every x epochs")
tf.app.flags.DEFINE_integer("refresh_hz", 100, "Reloading the browser every x epochs")
tf.app.flags.DEFINE_integer("update_target_network_hz", 20, "Reloading the browser every x epochs")
tf.app.flags.DEFINE_boolean("training", True, "Train a new model")
tf.app.flags.DEFINE_boolean("visualize", True, "Visualize")
FLAGS = tf.app.flags.FLAGS


def check_path_validity():
    """returns -1 if an unvalid path was given."""

    if FLAGS.training and os.path.exists(FLAGS.logdir):
        print("PATH FOR STORING RESULTS ALREADY EXISTS - Results would be overwritten.")
        return -1

    elif not FLAGS.training and not os.path.exists(FLAGS.logdir):
        print("PATH DOES NOT EXISTS. TRAINED MODEL NOT FOUND.")
        return -1

    return 0


def setup_summary():
    with tf.variable_scope("statistics"):
        summary_scalars = ["exploration", "ep_steps", "ep_reward"]
        summary_placeholders, summary_ops = {}, {}
        for tag in summary_scalars:
            summary_placeholders[tag] = tf.placeholder('float32', None)
            summary_ops[tag]  = tf.summary.scalar(tag, summary_placeholders[tag])
    return summary_ops, summary_placeholders


def summarize(session, writer, summary_ops, summary_placeholders, cnt, values):
    ops = [summary_ops[tag] for tag in list(values.keys())]
    feed_dict = {summary_placeholders[tag]: values[tag] for tag in list(values.keys())}
    summary_lists = session.run(ops, feed_dict)
    for summary in summary_lists:
        writer.add_summary(summary, cnt)


def play(agent, env, preprocessor):
    # load pretrained model,
    # will fail if the given path doesn't hold a valid model
    name = FLAGS.logdir + "rex.ckpt"
    if FLAGS.checkpoint_nr is not None:
        name = name + "-" + str(FLAGS.checkpoint_nr)

    agent.load(name)
    agent.explore_prob = 0.0

    while True:
        frame, _, crashed = env.start_game()
        frame = preprocessor.process(frame)
        state = preprocessor.get_initial_state(frame)

        while not crashed:
            action, _  = agent.act(state)
            next_frame, reward, crashed = env.do_action(action)
            print("action: {}".format(env.actions[action]))
            next_frame = preprocessor.process(next_frame)
            next_state = preprocessor.get_updated_state(next_frame)

            state = next_state

        print("Crash")

def train(agent, env, preprocessor, summarize_function):
    agent.update_target_network()

    epoch = 0
    while True:
        epoch += 1
        print("\nEpoch: ", epoch)

        frame, _ , crashed = env.start_game()
        frame = preprocessor.process(frame)
        state = preprocessor.get_initial_state(frame)
        ep_steps, ep_reward = 0, 0

        while not crashed:

            action, explored = agent.act(state)
            next_frame, reward, crashed = env.do_action(action)
            # A '*' is appended to the action if it was randomly chosen (i.e. not produced by the network)
            action_str = Environment.actions[action] + ["", "*"][explored]
            print("action: {}\t crashed: {}".format(action_str, crashed))
            next_frame = preprocessor.process(next_frame)
            next_state = preprocessor.get_updated_state(next_frame)
            agent.remember(state, action, reward, next_state, crashed)

            ep_steps += 1
            ep_reward += reward

            state = next_state

        agent.replay(epoch)
        agent.explore_less()

        if epoch % FLAGS.update_target_network_hz == 0:
            agent.update_target_network()

        stats = {"exploration": agent.explore_prob, "ep_steps": ep_steps, "ep_reward": ep_reward}
        summarize_function(epoch, stats)

        if epoch % FLAGS.checkpoint_hz == 0:
            agent.save(epoch)

        if epoch % FLAGS.refresh_hz == 0:
            env.refresh_game()


def main(_):

    if check_path_validity() == -1:
        exit(1)

    FLAGS.logdir = FLAGS.logdir if FLAGS.logdir.endswith('/') else FLAGS.logdir + '/'
    # Make a new directory to store checkpoints and tensorboard summaries,
    # this is only necessary if were are going to train a new model.
    if FLAGS.training:
        os.makedirs(FLAGS.logdir)

    # Setup tensorflow and tensorboard writers
    tf.reset_default_graph()
    session = tf.Session()
    writer = tf.summary.FileWriter(FLAGS.logdir, session.graph) if FLAGS.visualize else None
    summary_ops, summary_placeholders = setup_summary()

    # Initialize key objects: environment, agent and preprocessor
    env = Environment("127.0.0.1", 9090)
    agent = DDQNAgent(session, num_actions, width, height, FLAGS.logdir, writer)
    preprocessor = Preprocessor(width, height)

    if FLAGS.training:
        summarize_func = partial(summarize, session, writer, summary_ops, summary_placeholders)
        train(agent, env, preprocessor, summarize_func)
    else:
        play(agent, env, preprocessor)


if __name__ == '__main__':
    tf.app.run()
