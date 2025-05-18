from unityagents import UnityEnvironment
import numpy as np
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from maddpg_agent import MultiAgentDDPG
import numpy as np

class TennisAgentsVisualizer:
    def __init__(self, file_path: str = 'Tennis_Linux/Tennis.x86_64',
                 path_agents: str = 'models/Prob_Noise_DeepNet_BufferPlus_Learn_FreqMin_BatchNorm_03',
                 seed: int = 2):

        self.env = UnityEnvironment(file_name=file_path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.path_agents = path_agents
        # Reset to get environment specs
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.state_size = env_info.vector_observations.shape[1]
        self.action_size = self.brain.vector_action_space_size

        # Initialize agent and load weights
        self.multiagent = MultiAgentDDPG(state_size=self.state_size, action_size=self.action_size, random_seed=seed, num_agents=self.num_agents)
        self.load_models()

    def load_models(self):
        path = self.path_agents
        for i, agent in enumerate(self.multiagent.agents):
            actor_path = f'{path}/agent_{i}_actor.pth'
            critic_path = f'{path}/agent_{i}_critic.pth'

            # Load local networks
            agent.actor_local.load_state_dict(torch.load(actor_path))
            agent.critic_local.load_state_dict(torch.load(critic_path))

            # Sync targets manually
            agent.actor_target.load_state_dict(agent.actor_local.state_dict())
            agent.critic_target.load_state_dict(agent.critic_local.state_dict())

            # Switch to eval mode
            agent.actor_local.eval()
            agent.actor_target.eval()

            print(f"Loaded agent {i} models from {path}")

    def run(self, max_t: int = 1000):
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        states = env_info.vector_observations
        self.multiagent.reset()  # Reset noise/state if needed
        scores = np.zeros(self.num_agents)

        while True:
            actions = self.multiagent.act(states, add_noise=False)  
            env_info = self.env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            scores += rewards
            states = next_states

            if np.any(dones):
                break

        print(f"Final scores: Agent 0 = {scores[0]:.2f}, Agent 1 = {scores[1]:.2f}")
        self.env.close()

if __name__ == "__main__":
    visualizer = TennisAgentsVisualizer(file_path = 'Tennis_Linux/Tennis.x86_64',
                                        path_agents='models/Prob_Noise_DeepNet_BufferPlus_Learn_FreqMin_BatchNorm_03',
                 seed = 2)
    visualizer.run(max_t = 2000)
