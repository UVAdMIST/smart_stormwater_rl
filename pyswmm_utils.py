"""
Helper functions for controlling SWMM simulations with RL

Written by Benjamin Bowes, May 6, 2019
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """
    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) *\
            np.random.normal(size=self.size)
        self.x0 = x
        return x


def gen_noise(num_episode, act_space, mu=0.0, sigma=1):
    # generate an additive noise signal for action space exploration
    if num_episode > 0:
        sigma = sigma/np.sqrt(num_episode)
    if sigma < 0.01:  # this occurs after 10,000 episodes
        sigma = 0.01

    sample = np.random.normal(mu, sigma, act_space)

    return sample


def get_control_structures(inp_file, structure_type):
    """
    returns list of swmm structures to be controlled

    inp_file = path to swmm input file
    structure_type = section of input file to search for (ex. "[ORIFICES]")
    """
    start_line = None
    end_line = None

    with open(inp_file, 'r') as tmp_file:
        lines = tmp_file.readlines()

    struct_list = []
    for i, j in enumerate(lines):
        if j.startswith(structure_type):
            start_line = i
            for k, l in enumerate(lines[i + 1:]):
                if l.startswith("["):
                    end_line = k + i
                    break
            if not end_line:
                end_line = len(lines)

    for m in range(start_line + 3, end_line, 1):
        struct = lines[m].split(" ")[0]
        struct_list.append(struct)

    return struct_list


def save_state(out_lists, path):  # TODO dynamically set columns
    # saves state for plotting depths and flooding
    cols = ["St1_depth", "St2_depth", "J3_depth", "St1_flooding", "St2_flooding", "J3_flooding"]
    out_df = pd.DataFrame(out_lists).transpose()
    out_df.columns = cols
    out_df.to_csv(path, index=False)


def save_action(out_lists, path):
    # saves action for plotting policy
    cols = ["R1", "R2"]
    out_df = pd.DataFrame(out_lists).transpose()
    out_df.columns = cols
    out_df.to_csv(path, index=False)
