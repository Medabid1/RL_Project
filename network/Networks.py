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

class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, env_params['action'])
        self.apply(init_weights)
    def forward(self, x, goal):
        if not isinstance(x, torch.Tensor) :
            x = to_tensor(x)
        if not isinstance(goal, torch.Tensor) :
                goal = to_tensor(goal)
        x = torch.cat((x, goal), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.action_out(x)
        actions = self.max_action * torch.tanh(x)

        return (actions,x) 

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)
        self.apply(init_weights)
    def forward(self, x, actions, goals):
        if not isinstance(x, torch.Tensor) :
            x = to_tensor(x)
        if not isinstance(actions, torch.Tensor) :
                actions = to_tensor(actions)
        if not isinstance(goals, torch.Tensor) :
                goals = to_tensor(goals)
        x = torch.cat([x, goals, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value