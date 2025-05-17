import numpy as np
import random
import copy
from collections import namedtuple, deque

from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

from config_loader import Configurations

class MultiAgentDDPG():
    def __init__(self, num_agents, state_size, action_size, random_seed):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Configurations()
        
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        self.BUFFER_SIZE = config.hyperparameters.BUFFER_SIZE_MADDPG
        self.BACTH_SIZE = config.hyperparameters.BATCH_SIZE_MADDPG

    def act():
        pass

    def step():
        pass

    def reset():
        pass
