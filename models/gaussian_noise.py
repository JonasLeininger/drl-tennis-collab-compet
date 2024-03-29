import numpy as np
import random
import copy


class GaussianNoise():

    def __init__(self, size, mu=0., theta=0.15, sigma=0.15):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        # self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        noise = np.random.normal(self.mu, self.sigma, size=self.size)
        return noise
