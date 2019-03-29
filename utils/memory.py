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
                                    ['obs', 'action', 'reward', 'next_state', 'done', 'goal'])

    def add(self, obs, action, reward, next_state, done, goal, achieved_goal):
        e = self.experience(obs, action.detach().numpy(), reward, next_state, done, goal)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)

        observations = torch.Tensor(np.vstack([e.obs for e in experiences if e is not None])).float()
        actions = torch.Tensor(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.Tensor(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.Tensor(
                np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.Tensor(
                np.vstack([(1-e.done) for e in experiences if e is not None]).astype(np.uint8)).float()
        goals = torch.Tensor(np.vstack([e.goal for e in experiences if e is not None])).float()
        return (observations, actions, rewards, next_states, dones, goals)

    def __len__(self):
        return len(self.memory)



class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['states', 'actions', 'rewards', 'mask',
                       'values', 'q', 'pi', 'log_pi', 'entropy',
                       'advantages', 'returns', 'q_a', 'log_prob',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            assert k in self.keys
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)



