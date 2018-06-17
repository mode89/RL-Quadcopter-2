from collections import deque
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

def build_target_network_updates(targetNetwork, network, tau):
    return [targetParam.assign(param * tau + targetParam * (1.0 - tau))
        for targetParam, param in \
            zip(targetNetwork.params, network.params)]

class Network:

    pass

class ActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.network = self.build_network()
        self.targetNetwork = self.build_network()
        self.update_target_network_params = build_target_network_updates(
            self.targetNetwork, self.network, self.tau)
        self.build_network_updates()

    def build_network(self):
        inputs = tf.layers.Input(shape=[self.s_dim])

        initializer = tf.random_uniform_initializer(-0.003, 0.003)
        layers = [
            tf.layers.Dense(400, activation=tf.nn.relu),
            tf.layers.BatchNormalization(),
            tf.layers.Dense(300, activation=tf.nn.relu),
            tf.layers.BatchNormalization(),
            tf.layers.Dense(self.a_dim, activation=tf.tanh,
                kernel_initializer=initializer)
        ]

        outputs = inputs
        for layer in layers:
            outputs = layer(outputs)
        outputs = tf.multiply(outputs, self.action_bound)

        params = list()
        for layer in layers:
            params += layer.trainable_variables

        network = Network()
        network.inputs = inputs
        network.outputs = outputs
        network.params = params

        return network

    def build_network_updates(self):
        self.actionValueGradient = \
            tf.placeholder(tf.float32, [None, self.a_dim])
        gradients = tf.gradients(
            self.network.outputs,
            self.network.params,
            -self.actionValueGradient)
        gradients = map(lambda x: x / self.batch_size, gradients)
        self.networkUpdates = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(gradients, self.network.params))

    def train(self, inputs, a_gradient):
        self.sess.run(self.networkUpdates, feed_dict={
            self.network.inputs: inputs,
            self.actionValueGradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.network.outputs, feed_dict={
            self.network.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.targetNetwork.outputs, feed_dict={
            self.targetNetwork.inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.network = self.build_network()
        self.targetNetwork = self.build_network()
        self.update_target_network_params = build_target_network_updates(
            self.targetNetwork, self.network, self.tau)

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.reduce_mean(tf.square(
            self.predicted_q_value - self.network.outputs))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(
            self.network.outputs, self.network.actionInputs)

    def build_network(self):
        layers = list()

        stateInputs = tf.layers.Input(shape=[self.s_dim])
        layers.append(tf.layers.Dense(400, activation=tf.nn.relu))
        outputs = layers[-1](stateInputs)
        layers.append(tf.layers.BatchNormalization())
        outputs = layers[-1](outputs)
        layers.append(tf.layers.Dense(300))
        outputs1 = layers[-1](outputs)

        actionInputs = tf.layers.Input(shape=[self.a_dim])
        layers.append(tf.layers.Dense(300))
        outputs2 = layers[-1](actionInputs)

        outputs = tf.nn.relu(outputs1 + outputs2)
        initializer = tf.random_uniform_initializer(-0.003, 0.003)
        layers.append(tf.layers.Dense(1, kernel_initializer=initializer))
        outputs = layers[-1](outputs)

        params = list()
        for layer in layers:
            params += layer.trainable_variables

        network = Network()
        network.stateInputs = stateInputs
        network.actionInputs = actionInputs
        network.outputs = outputs
        network.params = params

        return network

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.network.outputs, self.optimize],
            feed_dict={
                self.network.stateInputs: inputs,
                self.network.actionInputs: action,
                self.predicted_q_value: predicted_q_value
            })

    def predict(self, inputs, action):
        return self.sess.run(self.network.outputs, feed_dict={
            self.network.stateInputs: inputs,
            self.network.actionInputs: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.targetNetwork.outputs, feed_dict={
            self.targetNetwork.stateInputs: inputs,
            self.targetNetwork.actionInputs: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.network.stateInputs: inputs,
            self.network.actionInputs: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                break

def main(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']))
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        train(sess, env, args, actor, critic, actor_noise)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
