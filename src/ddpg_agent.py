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
    
    def __init__(self, state_size, action_size, random_seed, agent_id, num_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            agent_id (int): ID of the agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.seed = random_seed + agent_id 
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed = self.seed, fc1_units=FC1_UNITS_ACTOR,fc2_units=FC1_UNITS_ACTOR, batch_norm=BATCH_NORMILIZATION_ACTOR).to(device)
        self.actor_target = Actor(state_size, action_size, seed = self.seed,fc1_units=FC1_UNITS_ACTOR,fc2_units=FC1_UNITS_ACTOR, batch_norm=BATCH_NORMILIZATION_ACTOR).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, num_agents = self.num_agents, seed = self.seed,fcs1_units=FC1_UNITS_CRITIC,fc2_units=FC1_UNITS_CRITIC, batch_norm=BATCH_NORMILIZATION_CRITIC).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents = self.num_agents,seed = self.seed,fcs1_units=FC1_UNITS_CRITIC,fc2_units=FC1_UNITS_CRITIC, batch_norm=BATCH_NORMILIZATION_CRITIC).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        ### OU Noise
        self.noise = OUNoise(action_size, random_seed, mu=MU, theta=THETA, sigma=SIGMA)

        ### Probabilistc Noise
        self.noise_scale = NOISE_INIT
        self.noise_decay = NOISE_DECAY
        self.min_noise = NOISE_MIN

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)

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

    def learn(
        self,
        agent_reward: torch.Tensor,
        agent_done: torch.Tensor,
        all_states: torch.Tensor,
        all_actions: torch.Tensor,
        all_next_states: torch.Tensor,
        all_next_actions: torch.Tensor,
        all_pred_actions: torch.Tensor,
        gamma: float,
        tau: float
    ):
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
        # ---------------------------- update critic ---------------------------- #
        # Target Q using target critic and target actors' actions
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            Q_targets_next = self.critic_target(all_next_states, all_next_actions)
            Q_targets = agent_reward + gamma * Q_targets_next * (1 - agent_done)

        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Addition for more stable training
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        self.actor_optimizer.zero_grad()
        # Compute actor loss
        actor_loss = -self.critic_local(all_states, all_pred_actions).mean()
        # Minimize the loss
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1) # Addition for more stable training
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau = tau)
        self.soft_update(self.actor_local, self.actor_target, tau = tau)                     

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




#    def step(self, state, action, reward, next_state, done, t):
#        """Save experience in replay memory, and use random sample from buffer to learn."""
#        # Save experience / reward
#
#        if REWARD_SCALING == True:
#            scaled_reward = reward * SCALE_FACTOR_REWARD
#            self.memory.add(state, action, scaled_reward, next_state, done)
#        else:
#            self.memory.add(state, action, reward, next_state, done)
#
#        t = t % LEARN_FREQ
#        # Learn, if enough samples are available in memory
#        if len(self.memory) > BATCH_SIZE and t == 0:
#            for _ in range(GRADIENT_UPDATES): # Including multiple updates sampled from memory
#                experiences = self.memory.sample()
#                self.learn(experiences, GAMMA)