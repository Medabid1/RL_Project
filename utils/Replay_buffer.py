import numpy as np
import random 
import torch

from copy import deepcopy
from torch import nn 
from collections import namedtuple, deque 

class ReplayBuffer():

    def __init__(self, act_size, buffer_size, batch_size):

        self.action_size = act_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names= 
                                    ['obs', 'action', 'reward', 'next_state', 'done'])

    def add(self, obs, action, reward, next_state, done):

        e = self.experience(obs, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)

        obss = torch.Tensor(np.vstack([e.obs for e in experiences if e is not None])).float()
        actions = torch.Tensor(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.Tensor(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.Tensor(
                np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.Tensor(
                np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (obss, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)