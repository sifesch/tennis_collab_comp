import numpy as np 

class ReplayBuffer:
    def __init__(self, max_size, cri_dim, act_dim, n_actions, n_agents, batch_size):
        self.memory_size = max_size
        self.memory_counter = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.act_dim = act_dim
        self.cri_dim = cri_dim

        self.state_memory = np.zeros((self.memory_size, self.cri_dim))
        self.new_state_memory = np.zeros((self.memory_size, self.cri_dim), dtype=bool)
        self.reward_memory = np.zeros((self.memory_size, n_agents))
        self.terminal_memory = np.zeros((self.memory_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.memory_size, self.act_dim[i]))
            )
            self.actor_new_state_memory.append(
                np.zeros((self.memory_size, self.act_dim[i]))
            )
            self.actor_action_memory.append(
                np.zeros((self.memory_size, self.n_actions))
            )
    
    def store_transitions(self, raw_observations, state, action, reward, raw_observations_, state_, done): 
        index = self.memory_counter % self.memory_size

        for agent_index in range(self.n_agents):
            self.actor_state_memory[agent_index][index] = raw_observations[agent_index]
            self.actor_new_state_memory[agent_index][index] = raw_observations_[agent_index]
            self.actor_action_memory[agent_index][index] = action[agent_index]

        self.state_memory[agent_index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def sample_buffer(self):