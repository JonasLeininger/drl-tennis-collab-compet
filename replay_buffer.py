import random
import numpy as np
from collections import namedtuple, deque

import torch

class ReplayBuffer(object):

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state1", "state2", "action1", "action2", "reward",
                                                                "next_state1", "next_state2", "done"])

    def add(self, state_1, state_2, action_1, action_2, rewards, next_state_1, next_state_2, dones):
        for state1, state2, action1, action2, reward, next_state1, next_state2, done in zip(state_1, state_2, action_1, action_2,
                                                           rewards, next_state_1, next_state_2, dones):
            e = self.experience(state1, state2, action1, action2, reward, next_state1, next_state2, done)
            self.memory.append(e)

    def sample(self):
        """
        Sample a minibatch from memory uniformly
        :return:
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        state1 = torch.from_numpy(np.vstack([e.state1 for e in experiences if e is not None])).float().to(self.device)
        state2 = torch.from_numpy(np.vstack([e.state2 for e in experiences if e is not None])).float().to(self.device)
        action1 = torch.from_numpy(np.vstack([e.action1 for e in experiences if e is not None])).float().to(self.device)
        action2 = torch.from_numpy(np.vstack([e.action2 for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_state1 = torch.from_numpy(np.vstack([e.next_state1 for e in experiences if e is not None])).float().to(self.device)
        next_state2 = torch.from_numpy(np.vstack([e.next_state2 for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (state1, state2, action1, action2, rewards, next_state1, next_state2, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)