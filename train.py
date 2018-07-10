from collections import deque
from agents.agent import Agent
import numpy as np
from task import Task
import tensorflow as tf
import json

def train(agent, task):
    rewards = deque(maxlen=100)
    summary = list()

    try:
        for episodeId in range(100000):
            episodeReward = 0.0
            state = task.reset()

            stepCount = 0
            while True:
                stepCount += 1
                action = agent.act(state)
                nextState, reward, done = task.step(action[0])
                agent.learn(nextState=nextState, reward=reward, done=done)
                state = nextState
                episodeReward += reward
                if done: break

            rewards.append(episodeReward)
            avgReward = np.mean(rewards)

            print("Episode: {} Reward: {:4.3f} Avg. reward: {:4.3f}". \
                format(episodeId, episodeReward, avgReward))

            summary.append({
                "episodeId": episodeId,
                "reward": episodeReward,
                "averageReward": avgReward,
            })
    finally:
        print("Dumping summary ...")
        with open("summary.json", "w") as summaryFile:
            json.dump(summary, summaryFile, indent=True)

if __name__ == "__main__":
    task = Task()

    agent = Agent(
        stateDim=task.state_size,
        actionDim=task.action_size,
        actionBound=[task.action_low, task.action_high],
        actorLearningRate=0.0001,
        criticLearningRate=0.001,
        batchSize=64,
        tau=0.01,
        gamma=0.99,
        bufferSize=1000000,
        seed=1234)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        agent.set_session(session)
        train(agent, task)
