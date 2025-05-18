import numpy as np 
import torch
import numpy as np
import random
from collections import namedtuple, deque
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, num_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.num_agents = num_agents

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new multi-agent experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory for multiple agents."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [[] for _ in range(self.num_agents)]
        actions = [[] for _ in range(self.num_agents)]
        rewards = [[] for _ in range(self.num_agents)]
        next_states = [[] for _ in range(self.num_agents)]
        dones = [[] for _ in range(self.num_agents)]

        for e in experiences:
            for i in range(self.num_agents):
                states[i].append(e.states[i])
                actions[i].append(e.actions[i])
                rewards[i].append(e.rewards[i])
                next_states[i].append(e.next_states[i])
                dones[i].append(e.dones[i])

        states = [torch.from_numpy(np.vstack(s)).float().to(device) for s in states]
        actions = [torch.from_numpy(np.vstack(s)).float().to(device) for s in actions]
        rewards = [torch.from_numpy(np.vstack(s)).float().to(device) for s in rewards]
        next_states = [torch.from_numpy(np.vstack(s)).float().to(device) for s in next_states]
        dones = [torch.from_numpy(np.vstack(s).astype(np.float32)).to(device) for s in dones]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)