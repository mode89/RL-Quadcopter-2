import numpy as np
from physics_sim import PhysicsSim

TARGET_POSE = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

class Task():

    def __init__(self):
        self.sim = PhysicsSim(init_pose=TARGET_POSE, runtime=1.0)
        self.state_size = self.state.shape[0]
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

    def get_reward(self):
        error = np.concatenate([
            self.get_position_error(),
            self.get_orientation_error()
        ])
        reward = -np.linalg.norm(error)
        return reward

    def get_position_reward(self):
        return -self.get_position_error().sum()

    def get_position_error(self):
        error = TARGET_POSE[:3] - self.sim.pose[:3]
        error = np.abs(error)
        return error

    def get_orientation_reward(self):
        return -self.get_orientation_error().sum()

    def get_orientation_error(self):
        error = TARGET_POSE[3:6] - self.sim.pose[3:6]
        error = np.remainder(error, 2.0 * np.pi)
        error[error > np.pi] -= 2.0 * np.pi
        error = np.abs(error)
        return error

    def get_accel_reward(self):
        direction = TARGET_POSE[:3] - self.sim.pose[:3]
        velocity = self.sim.v
        timeStep = self.sim.dt
        timeStep2 = timeStep * timeStep
        perfectAccel = 2.0 * (direction - velocity * timeStep) / timeStep2
        perfectAccelNorm = np.linalg.norm(perfectAccel)
        maxAccel = 50.0
        if perfectAccelNorm > maxAccel:
            perfectAccel = perfectAccel / perfectAccelNorm * maxAccel
            perfectAccelNorm = maxAccel
        accel = self.sim.linear_accel
        accelError = perfectAccel - accel
        reward = 1.0 - np.linalg.norm(accelError) / perfectAccelNorm
        return reward

    @property
    def state(self):
        return np.concatenate([
            self.sim.pose,
            self.sim.v,
            self.sim.angular_v,
            self.sim.linear_accel,
            self.sim.angular_accels
        ])

    def step(self, rotorSpeeds):
        done = self.sim.next_timestep(rotorSpeeds)
        reward = self.get_reward()
        nextState = self.state
        return nextState, reward, done

    def reset(self):
        self.sim.reset()
        return self.state
