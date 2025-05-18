import numpy as np
import random
import copy
from collections import namedtuple, deque

from replay_buffer import ReplayBuffer
from ddpg_agent import DDPGAgent

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

        self.BUFFER_SIZE = config.hyperparameters.BUFFER_SIZE
        self.BATCH_SIZE = config.hyperparameters.BATCH_SIZE
        self.GAMMA = config.hyperparameters.GAMMA
        self.TAU = config.hyperparameters.TAU
        self.LEARN_FREQ = config.hyperparameters.LEARN_FREQ
        self.GRADIENT_UPDATES = config.hyperparameters.GRADIENT_UPDATES
        self.t_step = 0

        self.noise_choice = config.noise_config.GENERAL.PROB_NOISE_OR_OU
        self.add_noise = config.noise_config.GENERAL.ADD_NOISE
        self.noise_weight = config.noise_config.ProbNoise_Config.NOISE_INIT
        self.noise_decay = config.noise_config.ProbNoise_Config.NOISE_DECAY
        self.t_stop_noise = config.noise_config.GENERAL.STOP_NOISE

        self.memory = ReplayBuffer(buffer_size = self.BUFFER_SIZE, 
                                   batch_size = self.BATCH_SIZE, 
                                   num_agents = num_agents
                                   )
        
        self.agents = [DDPGAgent(state_size=state_size, action_size=action_size, num_agents = self.num_agents, random_seed=self.seed, agent_id = i) for i in range(num_agents)]    

    def step(self, states, actions, rewards, next_states, dones) -> None:
        
        
        self.memory.add(states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states, 
                        dones=dones)

        if self.t_step > self.t_stop_noise:
            self.add_noise = False

        self.t_step = self.t_step + 1
        
        if len(self.memory) > self.BATCH_SIZE and self.t_step % self.LEARN_FREQ == 0:
            for _ in range(self.GRADIENT_UPDATES):
                self.learn()

    def reset(self) -> None:
        """Resets each agent
        """
        for agent in self.agents:
            agent.reset()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Concatenate states for critic input
        full_states = torch.cat(states, dim=1)
        full_next_states = torch.cat(next_states, dim=1)

        # Precompute all target next actions
        all_next_actions = [agent.actor_target(state) for agent, state in zip(self.agents, next_states)]
        full_next_actions = torch.cat(all_next_actions, dim=1)

        # Precompute current actions from replay buffer
        full_actions = torch.cat(actions, dim=1)

        for i, agent in enumerate(self.agents):
            agent_state = states[i]
            agent_action = actions[i]
            agent_reward = rewards[i] 
            agent_next_state = next_states[i]
            agent_done = dones[i]    

            predicted_agent_action = agent.actor_local(agent_state)

            all_actions_pred = [a.clone().detach() for a in actions]
            all_actions_pred[i] = predicted_agent_action
            full_predicted_actions = torch.cat(all_actions_pred, dim=1)

            agent.learn(
                agent_reward=agent_reward,
                agent_done=agent_done,
                all_states=full_states,
                all_actions=full_actions,
                all_next_states=full_next_states,
                all_next_actions=full_next_actions,
                all_pred_actions=full_predicted_actions,
                gamma=self.GAMMA,
                tau=self.TAU
            )

    def act(self, states: list, add_noise: bool) -> np.ndarray:
        '''Get actions from all agents given their states

        Args:
            states (list): list of states
            add_noise (bool, optional): boolean wheter to add nosie. Defaults to True.

        Raises:
            ValueError: In case wrong noise type was defined in configurations

        Returns:
            np.ndarray: actions that agents will take given their states
        '''
        actions = [] 


        for i, agent in enumerate(self.agents):
            if self.noise_choice=="ou":
                action = agent.act_ou(states[i], add_noise = add_noise)
            elif self.noise_choice == "prob":
                action = agent.act_probabilistic(states[i], add_noise = add_noise)
            else:
                raise ValueError("Unsupported noise type. Choose 'ou' or 'prob'.")
            actions.append(action)
        return np.stack(actions)
