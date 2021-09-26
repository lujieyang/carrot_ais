import os
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from env.carrot_sim import CarrotSim
from env.carrot_rewards import lyapunov, image_transform
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import process_obs


class CarrotEnv(gym.Env):
    metadata = {'render:modes': ['human']}

    def __init__(self):
        self.sim = CarrotSim()
        self.episode_length = 50
        self.episode_step = 0
        self.observation_dim = 3*32*32
        action_table = []
        x_grid = 11
        y_grid = 4
        for i in range(x_grid):
            for j in range(y_grid):
                for k in range(x_grid):
                    for l in range(y_grid):
                        if (i, j) != (k, l) and np.abs(i - k) + np.abs(j - l) <= 3:
                            action_table.append(
                                [
                                    -0.5 + 1 / (x_grid - 1) * i,
                                    -0.3 + 0.6 / (y_grid - 1) * j,
                                    -0.5 + 1 / (x_grid - 1) * k,
                                    -0.3 + 0.6 / (y_grid - 1) * l,
                                ]
                            )
        # num_grid = 4
        # for i in range(num_grid):
        #     for j in range(num_grid):
        #         for k in range(num_grid):
        #             for l in range(num_grid):
        #                 if (i, j) != (k, l) and (i-k)**2 + (j-l)**2 <= (num_grid-1)**2/2:
        #                     action_table.append(
        #                         [
        #                             -0.4 + 0.8 / (num_grid - 1) * i,
        #                             -0.4 + 0.8 / (num_grid - 1) * j,
        #                             -0.4 + 0.8 / (num_grid - 1) * k,                                                
        #                             -0.4 + 0.8 / (num_grid - 1) * l,                        
        #                             ]
        #                         )
        self.action_table = np.array(action_table)
        self.action_space = spaces.Discrete(self.action_table.shape[0])
        self.observation_space = spaces.Box(
            low=0,
            high=1, shape=(self.observation_dim,),
            dtype=np.float32
        )

    def step(self, action_ind):
        done = False
        action = self.action_table[action_ind]
        self.sim.update(action)
        next_image = image_transform(self.sim.get_current_image())
        reward = lyapunov(next_image)
        obs = process_obs(255.0 * next_image)
        self.episode_step +=  1
        if self.episode_step == self.episode_length:
            done = True
        return obs, -reward, done, {}

    def reset(self):
        self.episode_step = 0
        self.sim.refresh()
        img = image_transform(self.sim.get_current_image())
        obs = process_obs(img)
        return obs

    def render(self, mode='human'):
        # Return full resolution image for debugging / rendering.
        return 255.0 * image_transform(self.sim.get_current_image(), size=(256, 256))

    def get_reward(self, goal='square'):
        current_image = image_transform(self.sim.get_current_image(), size=(32, 32))
        reward = lyapunov(current_image)
        return reward

    def close(self):
        pass


