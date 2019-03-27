import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from utils.utils import init_weights, to_tensor

class ActorCriticContinous(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, std=0.0):
        super(ActorCriticContinous, self).__init__()
         
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
           
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
            state = to_tensor(state)

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

class ActorCriticDeterministic(nn.Module):
    def __init__(self, state_size, action_size, goal_size, hidden_layers, use_her=False):
        super(ActorCriticDeterministic, self).__init__()
        self.input_dim = state_size 
        if use_her :
            self.input_dim += goal_size 

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim + action_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1],1)
        )
            
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], action_size),
            nn.Tanh()
        )
        
        
        self.apply(init_weights)

    def forward_critic(self, state, action, goal=None):
        if not isinstance(state, torch.Tensor) :
            state = to_tensor(state)
        if not isinstance(action, torch.Tensor) :
            action = to_tensor(action)
        if goal is not None :
            if not isinstance(goal, torch.Tensor) :
                goal = to_tensor(goal)
            x = torch.cat((state, action, goal), 1)
        else:    
            x = torch.cat((state, action), 1)
        return self.critic(x)
    
    def forward_actor(self, state, goal=None):
        if not isinstance(state, torch.Tensor) :
            state = to_tensor(state)
        if goal is not None :
            if not isinstance(goal, torch.Tensor) :
                goal = to_tensor(goal)
            x = torch.cat((state, goal), -1)
        else :
            x = state
        return self.actor(x)
