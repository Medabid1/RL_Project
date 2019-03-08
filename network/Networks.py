import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 


class Actor(nn.Module):
    def __init__(self, obs_size, act_size, hidden_layers):
        super(Actor, self).__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(obs_size, hidden_layers[0])])
        self.sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for (h1, h2) in self.sizes ])
        
        self.mu = nn.Linear(hidden_layers[-1], act_size)
        self.sigma = nn.Linear(hidden_layers[-1], 1)

        self.relu = nn.ReLU()
    
    def forward(self, obs):
        x = torch.from_numpy(obs)
        for linear in self.hidden_layers : 
            x = self.relu(linear(x))
        
        mu = self.mu(x)
        sigma = F.softplus(self.sigma)
        self.policy = torch.distributions.Normal(mu, sigma)
        sample = torch.clamp(self.policy.sample_n(1), -1, 1)

        return sample


class Critic(nn.Module):
    def __init__(self, obs_size, act_size, hidden_layers):
        super(Critic, self).__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(obs_size, hidden_layers[0])])
        self.sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for (h1, h2) in self.sizes ])
        self.output = nn.Linear(hidden_layers[-1], act_size)
        self.relu = nn.ReLU()

    def forward(self, obs):
        x = torch.form_numpy(obs)
        for linear in self.hidden_layers : 
            x = self.relu(linear(x))
        
        return self.output(x)

class 