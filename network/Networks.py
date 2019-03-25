import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from utils.utils import init_weights

class ActorCriticContinous(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, std=0.0):
        super(ActorCriticContinous, self).__init__()
         
        self.critic = nn.ModuleList([
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ])
    
           
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(action_size) * std)
        
        self.apply(init_weights)
        
    def forward(self, state, action=None):

        if not isinstance(state, torch.Tensor) :
            state = torch.from_numpy(state).float()

        value = self.critic(state)
        mu    = self.actor(state)
        std   = self.log_std
        dist  = torch.distributions.Normal(mu, F.softplus(std))
        if action is None :
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'actions': action.unsqueeze(0),
                'log_prob': log_prob,
                'entropy': entropy,
                'mean': mu,
                'values': value}

class DDPGActorCritic(nn.Module):

    def __init__(self, state_size, action_size, hidden_size, std=0.0):
        super(DDPGActorCritic, self).__init__() 
        self.critic = nn.Sequential(
            nn.Linear(state_size+action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1)
            )
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(action_size) * std)
        
        self.apply(init_weights)
    
    def actor_forward(self, state, action=None):
        
        if not isinstance(state, torch.Tensor) :
            state = torch.from_numpy(state).float()
        mu    = self.actor(state)
        std   = self.log_std
        dist  = torch.distributions.Normal(mu, F.softplus(std))    
        if action is None :
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        return {'actions': action.unsqueeze(0),
                'log_prob': log_prob,
                'mean': mu
                }

    def critic_forward(self, state, action):
        inputs = torch.cat((state, action), dim=0)
        return self.critic(inputs)
