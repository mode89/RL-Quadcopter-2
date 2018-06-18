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

    def predict(self, session, *inputs):
        return session.run(
            self.outputs,
            feed_dict=dict(zip(self.inputs, inputs)))

class Actor:

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
        network.inputs = [inputs]
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

    def train(self, inputs, actionValueGradient):
        self.sess.run(self.networkUpdates, feed_dict={
            self.network.inputs[0]: inputs,
            self.actionValueGradient: actionValueGradient
        })

    def predict(self, inputs):
        return self.network.predict(self.sess, inputs)

    def predict_target(self, inputs):
        return self.targetNetwork.predict(self.sess, inputs)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class Critic:

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
        self.build_network_updates()

        self.actionValueGradient = tf.gradients(
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
        network.inputs = [stateInputs, actionInputs]
        network.stateInputs = network.inputs[0]
        network.actionInputs = network.inputs[1]
        network.outputs = outputs
        network.params = params

        return network

    def build_network_updates(self):
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        loss = tf.reduce_mean(tf.square(
            self.predicted_q_value - self.network.outputs))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(loss)

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.network.outputs, self.optimize],
            feed_dict={
                self.network.stateInputs: inputs,
                self.network.actionInputs: action,
                self.predicted_q_value: predicted_q_value
            })

    def predict(self, inputs, action):
        return self.network.predict(self.sess, inputs, action)

    def predict_target(self, inputs, action):
        return self.targetNetwork.predict(self.sess, inputs, action)

    def action_value_gradient(self, inputs, actions):
        return self.sess.run(self.actionValueGradient, feed_dict={
            self.network.stateInputs: inputs,
            self.network.actionInputs: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class OUNoise:

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3, dt=0.01):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + \
            self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state

class Agent:

    def __init__(self, session, stateDim, actionDim, actionMin, actionMax,
            actorLearningRate, criticLearningRate, batchSize, tau, gamma,
            bufferSize, seed):
        self.batchSize = batchSize
        self.actor = Actor(sess=session, state_dim=stateDim,
            action_dim=actionDim, action_bound=(actionMax - actionMin) / 2.0,
            learning_rate=actorLearningRate, tau=tau, batch_size=batchSize)
        self.critic = Critic(sess=session, state_dim=stateDim,
            action_dim=actionDim, learning_rate=criticLearningRate,
            tau=tau, gamma=gamma)
        self.actorNoise = OUNoise(actionDim)
        self.replayBuffer = ReplayBuffer(bufferSize, seed)

    def act(self, state):
        self.lastState = state
        self.lastAction = \
            self.actor.predict(np.reshape(state, (1, self.actor.s_dim))) + \
            self.actorNoise.sample()
        return self.lastAction

    def learn(self, nextState, reward, done):
        self.replayBuffer.add(
            np.reshape(self.lastState, (self.actor.s_dim,)),
            np.reshape(self.lastAction, (self.actor.a_dim,)),
            reward,
            done,
            np.reshape(nextState, (self.actor.s_dim,)))

        if self.replayBuffer.size() > self.batchSize:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                self.replayBuffer.sample_batch(self.batchSize)

            # Calculate targets
            target_q = self.critic.predict_target(
                s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(self.batchSize):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(
                s_batch, a_batch, np.reshape(y_i, (self.batchSize, 1)))

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_value_gradient(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

def train(env, agent, args):
    # Initialize target network weights
    agent.actor.update_target_network()
    agent.critic.update_target_network()

    for i in range(int(args['max_episodes'])):
        ep_reward = 0
        state = env.reset()
        for j in range(int(args['max_episode_len'])):
            if args['render_env']:
                env.render()

            action = agent.act(state)
            nextState, reward, done, info = env.step(action[0])
            agent.learn(nextState, reward, done)
            state = nextState
            ep_reward += reward

            if done:
                print("Episode: {} Reward: {:7.3f}".format(i, ep_reward))
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

        agent = Agent(
            session=sess,
            stateDim=state_dim,
            actionDim=action_dim,
            actionMin=env.action_space.low,
            actionMax=env.action_space.high,
            actorLearningRate=args["actor_lr"],
            criticLearningRate=args["critic_lr"],
            batchSize=args["minibatch_size"],
            tau=args["tau"],
            gamma=args["gamma"],
            bufferSize=args["buffer_size"],
            seed=args["random_seed"])

        sess.run(tf.global_variables_initializer())

        train(env, agent, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr',
        help='actor network learning rate',
        type=float,
        default=0.0001)
    parser.add_argument('--critic-lr',
        help='critic network learning rate',
        type=float,
        default=0.001)
    parser.add_argument('--gamma',
        help='discount factor for critic updates',
        type=float,
        default=0.99)
    parser.add_argument('--tau', help='soft target update parameter',
        type=float,
        default=0.001)
    parser.add_argument('--buffer-size',
        help='max size of the replay buffer',
        type=int,
        default=1000000)
    parser.add_argument('--minibatch-size',
        help='size of minibatch for minibatch-SGD',
        type=int,
        default=64)

    # run parameters
    parser.add_argument('--env',
        help='choose the gym env- tested on {Pendulum-v0}',
        default='Pendulum-v0')
    parser.add_argument('--random-seed',
        help='random seed for repeatability',
        type=int,
        default=1234)
    parser.add_argument('--max-episodes',
        help='max num of episodes to do while training',
        type=int,
        default=50000)
    parser.add_argument('--max-episode-len',
        help='max length of 1 episode',
        type=int,
        default=1000)
    parser.add_argument('--render-env',
        help='render the gym env',
        action='store_true')
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
