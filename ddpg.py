from collections import deque
import tensorflow as tf
import numpy as np
import gym
import argparse
import random

class ReplayBuffer(object):

    class Sample:

        def __init__(self, state, action, reward, nextState, done):
            self.state = state
            self.action = action
            self.reward = [reward]
            self.nextState = nextState
            self.done = [1.0 if done else 0.0]

    class Batch:

        def __init__(self, samples):
            self.size = len(samples)
            self.state = np.array([s.state for s in samples])
            self.action = np.array([s.action for s in samples])
            self.reward = np.array([s.reward for s in samples])
            self.nextState = np.array([s.nextState for s in samples])
            self.done = np.array([s.done for s in samples])

    def __init__(self, bufferSize, batchSize, seed=123):
        self.buffer = deque(maxlen=bufferSize)
        self.batchSize = batchSize
        self.random = random.Random()
        self.random.seed(seed)

    def add(self, state, action, reward, nextState, done):
        sample = ReplayBuffer.Sample(state, action, reward, nextState, done)
        self.buffer.append(sample)

    def sample(self):
        if len(self.buffer) < self.batchSize:
            return None
        else:
            samples = self.random.sample(self.buffer, self.batchSize)
            return ReplayBuffer.Batch(samples)

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

    def __init__(self,
            state_dim,
            action_dim,
            action_bound,
            learning_rate,
            tau,
            batch_size):
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
            tf.layers.Dense(1024, activation=tf.nn.relu),
            tf.layers.BatchNormalization(),
            tf.layers.Dense(1024, activation=tf.nn.relu),
            tf.layers.BatchNormalization(),
            tf.layers.Dense(self.a_dim, activation=tf.tanh,
                kernel_initializer=initializer)
        ]

        outputs = inputs
        for layer in layers:
            outputs = layer(outputs)

        actionMin = self.action_bound[0]
        actionMax = self.action_bound[1]
        actionRangeHalf = (actionMax - actionMin) / 2.0
        outputs = outputs * actionRangeHalf + (actionMin + actionRangeHalf)

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
        self.session.run(self.networkUpdates, feed_dict={
            self.network.inputs[0]: inputs,
            self.actionValueGradient: actionValueGradient
        })

    def predict(self, inputs):
        return self.network.predict(self.session, inputs)

    def predict_target(self, inputs):
        return self.targetNetwork.predict(self.session, inputs)

    def update_target_network(self):
        self.session.run(self.update_target_network_params)

class Critic:

    def __init__(self,
            state_dim,
            action_dim,
            learning_rate,
            tau,
            gamma):
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
        actionInputs = tf.layers.Input(shape=[self.a_dim])

        layers.append(tf.layers.Dense(1024, activation=tf.nn.relu))
        stateOutputs = layers[-1](stateInputs)
        layers.append(tf.layers.BatchNormalization())
        stateOutputs = layers[-1](stateOutputs)
        layers.append(tf.layers.Dense(1024))
        stateOutputs = layers[-1](stateOutputs)
        layers.append(tf.layers.BatchNormalization())
        stateOutputs = layers[-1](stateOutputs)

        layers.append(tf.layers.Dense(1024))
        actionOutputs = layers[-1](actionInputs)
        layers.append(tf.layers.BatchNormalization())
        actionOutputs = layers[-1](actionOutputs)

        outputs = tf.nn.relu(stateOutputs + actionOutputs)

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
        return self.session.run([self.network.outputs, self.optimize],
            feed_dict={
                self.network.stateInputs: inputs,
                self.network.actionInputs: action,
                self.predicted_q_value: predicted_q_value
            })

    def predict(self, inputs, action):
        return self.network.predict(self.session, inputs, action)

    def predict_target(self, inputs, action):
        return self.targetNetwork.predict(self.session, inputs, action)

    def action_value_gradient(self, inputs, actions):
        return self.session.run(self.actionValueGradient, feed_dict={
            self.network.stateInputs: inputs,
            self.network.actionInputs: actions
        })

    def update_target_network(self):
        self.session.run(self.update_target_network_params)

class OUNoise:

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3, dt=0.01):
        self.random = np.random.RandomState(seed=1234)
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
            self.sigma * self.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state

class Agent:

    def __init__(self,
            stateDim,
            actionDim,
            actionBound,
            actorLearningRate,
            criticLearningRate,
            batchSize,
            tau,
            gamma,
            bufferSize,
            seed):
        self.actor = Actor(
            state_dim=stateDim,
            action_dim=actionDim,
            action_bound=actionBound,
            learning_rate=actorLearningRate,
            tau=tau,
            batch_size=batchSize)
        self.critic = Critic(
            state_dim=stateDim,
            action_dim=actionDim,
            learning_rate=criticLearningRate,
            tau=tau,
            gamma=gamma)
        self.actorNoise = OUNoise(actionDim)
        self.replayBuffer = ReplayBuffer(
            bufferSize=bufferSize,
            batchSize=batchSize,
            seed=seed)

    def set_session(self, session):
        self.actor.session = session
        self.critic.session = session

    def act(self, state):
        self.lastState = state
        self.lastAction = \
            self.actor.predict(np.reshape(state, (1, self.actor.s_dim))) + \
            self.actorNoise.sample()
        return self.lastAction

    def learn(self, nextState, reward, done):
        maxQ = 0.0

        self.replayBuffer.add(
            state=np.reshape(self.lastState, (self.actor.s_dim,)),
            action=np.reshape(self.lastAction, (self.actor.a_dim,)),
            reward=reward,
            done=done,
            nextState=np.reshape(nextState, (self.actor.s_dim,)))

        batch = self.replayBuffer.sample()
        if batch is not None:
            # Calculate targets
            target_q = self.critic.predict_target(
                batch.nextState,
                self.actor.predict_target(batch.nextState))

            y = batch.reward + np.multiply(
                1.0 - batch.done, self.critic.gamma * target_q)

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(
                batch.state, batch.action, y)
            maxQ = np.amax(predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(batch.state)
            grads = self.critic.action_value_gradient(batch.state, a_outs)
            self.actor.train(batch.state, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

        return maxQ

def train(env, agent, args):
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
    env = gym.make(args['env'])
    np.random.seed(int(args['random_seed']))
    tf.set_random_seed(int(args['random_seed']))
    env.seed(int(args['random_seed']))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(
        stateDim=state_dim,
        actionDim=action_dim,
        actionBound=[env.action_space.low, env.action_space.high],
        actorLearningRate=args["actor_lr"],
        criticLearningRate=args["critic_lr"],
        batchSize=args["minibatch_size"],
        tau=args["tau"],
        gamma=args["gamma"],
        bufferSize=args["buffer_size"],
        seed=args["random_seed"])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        agent.set_session(session)
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

    main(args)
