import numpy as np
import copy
from collections import namedtuple, deque
import random

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, min_sigma=0.05, decay=0.99):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.initial_sigma = sigma  # save for reset
        self.min_sigma = min_sigma
        self.decay = decay
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    def decay_sigma(self):
        """Decay the sigma (exploration noise scale)."""
        self.sigma = max(self.min_sigma, self.sigma * self.decay)

