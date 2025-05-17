import numpy as np
import random

from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

from config_loader import Configurations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = Configurations()

# GENERAL HYPERPARAMETERS
BUFFER_SIZE = config.hyperparameters.BUFFER_SIZE
BATCH_SIZE = config.hyperparameters.BATCH_SIZE
GAMMA = config.hyperparameters.GAMMA
TAU = config.hyperparameters.TAU
LEARN_FREQ = config.hyperparameters.LEARN_FREQ
GRADIENT_UPDATES =  config.hyperparameters.GRADIENT_UPDATES

# NOISE HYPERPARAMETERS
NOISE_DECAY = config.noise_config.ProbNoise_Config.NOISE_DECAY
NOISE_INIT = config.noise_config.ProbNoise_Config.NOISE_INIT
NOISE_MIN = config.noise_config.ProbNoise_Config.NOISE_MIN
MU = config.noise_config.OUNoise_Config.MU
SIGMA = config.noise_config.OUNoise_Config.SIGMA
THETA = config.noise_config.OUNoise_Config.THETA

# REWARD SCALING HYPERPARAMETERS
REWARD_SCALING = config.hyperparameters.REWARD_SCALING
SCALE_FACTOR_REWARD = config.hyperparameters.SCALE_FACTOR_REWARD

# ACTOR HYPERPARAMETERS
LR_ACTOR = float(config.hyperparameters.ACTOR_PARAMS.LR_ACTOR)
FC1_UNITS_ACTOR = config.hyperparameters.ACTOR_PARAMS.FC1_UNITS
FC2_UNITS_ACTOR = config.hyperparameters.ACTOR_PARAMS.FC2_UNITS
BATCH_NORMILIZATION_ACTOR = config.hyperparameters.ACTOR_PARAMS.BATCH_NORMILIZATION

# CRITIC HYPERPARAMETERS
LR_CRITIC = float(config.hyperparameters.CRITIC_PARAMS.LR_CRITIC)
WEIGHT_DECAY = config.hyperparameters.CRITIC_PARAMS.WEIGHT_DECAY
FC1_UNITS_CRITIC = config.hyperparameters.CRITIC_PARAMS.FC2_UNITS
FC2_UNITS_CRITIC = config.hyperparameters.CRITIC_PARAMS.FC2_UNITS
BATCH_NORMILIZATION_CRITIC = config.hyperparameters.CRITIC_PARAMS.BATCH_NORMILIZATION

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed,n_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed,fc1_units=FC1_UNITS_ACTOR,fc2_units=FC1_UNITS_ACTOR, batch_norm=BATCH_NORMILIZATION_ACTOR).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed,fc1_units=FC1_UNITS_ACTOR,fc2_units=FC1_UNITS_ACTOR, batch_norm=BATCH_NORMILIZATION_ACTOR).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed,fcs1_units=FC1_UNITS_CRITIC,fc2_units=FC1_UNITS_CRITIC, batch_norm=BATCH_NORMILIZATION_CRITIC).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed,fcs1_units=FC1_UNITS_CRITIC,fc2_units=FC1_UNITS_CRITIC, batch_norm=BATCH_NORMILIZATION_CRITIC).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        ### OU Noise
        self.noise = OUNoise(action_size, random_seed, mu=MU, theta=THETA, sigma=SIGMA)

        ### Probabilistc Noise
        self.noise_scale = NOISE_INIT
        self.noise_decay = NOISE_DECAY
        self.min_noise = NOISE_MIN

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        if REWARD_SCALING == True:
            scaled_reward = reward * SCALE_FACTOR_REWARD
            self.memory.add(state, action, scaled_reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)

        t = t % LEARN_FREQ
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t == 0:
            for _ in range(GRADIENT_UPDATES): # Including multiple updates sampled from memory
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act_ou(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def act_probabilistic(self, state, add_noise=True):
        """Returns actions for given state using a stochastic policy with Tanh-squashed Gaussian noise."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state)
        self.actor_local.train()
        if add_noise:
            action += torch.randn_like(action) * self.noise_scale
        return np.clip(action.cpu().data.numpy(), -1, 1)

    def noise_update(self):
        """noise_update"""
        self.noise_scale = max(self.noise_scale * self.noise_decay, self.min_noise)
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Addition for more stable training
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1) # Addition for more stable training
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)