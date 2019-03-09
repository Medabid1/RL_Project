import torch 
import torch.optim as optim
import numpy as np
from BaseAgent import BaseAgent
from network.Networks import Actor, Critic

class A2CAgent(BaseAgent):

    def __init__(self, config, obs_size, action_size):
        super(A2CAgent).__init__()
        self.config = config
        self.actor = Actor(obs_size, action_size, config.actor_layers)
        self.critic = Critic(obs_size, action_size, config.critic_layers)
        self.optimizer = optim.Adam([{'params':self.actor.parameters()},
                                    {'params':self.critic.parameters()}],
                                     lr = 1e-5)
    def act(self, obs) :
        
        return self.actor(obs)

    def sample_value(self, obs): 
        
        return self.critic(obs)

    def step(self, obs, action, reward, next_obs, done):
         
        if not done :
            v = self.sample_value(obs)
            
            next_value = self.sample_value(next_obs) 
            td_error = reward + self.config.discount * next_value - v
            value_loss = 0.5 * td_error.pow(2).mean()
            policy_loss = -(td_error * self.actor.policy.log_prob(action)).mean()
            self.optimizer.zero_grad()
            loss = policy_loss + value_loss
            loss.backward()
            self.optimizer.step()
    
