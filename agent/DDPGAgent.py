import torch 
import numpy as np
from utils.noise import OrnsteinUhlenbeckProcess
from network.Networks import *
from .BaseAgent import BaseAgent
from network.Networks import DDPGActorCritic
from utils.memory import ReplayBuffer 

class DDPGAgent(BaseAgent):
    def __init__(self, config, state_size, action_size, hidden_size, env): 
        super(DDPGAgent, self).__init__()
        self.network = DDPGActorCritic(state_size, action_size, hidden_size)
        self.target_network = DDPGActorCritic(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.network.state_dict())
        self.memory = ReplayBuffer(action_size, buffer_size=1e4, batch_size=64)
        self.actor_optimizer = torch.optim.Adam(self.network.parameters())
        self.critic_optimize = torch.optim.Adam(self.network)
        self.noise = OrnsteinUhlenbeckProcess(state_size, seed=445684)
        self.env = env
    
    def sample(self):
        state = self.env.reset()
        self.reset()
        while True :
            predictions = self.network(state)
            action = predictions['actions']
            if self.config.add_noise :
                action = predictions['actions'] + self.noise.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(state, action, reward, next_state, done)
            if done :
                break
   
    def step(self):
        if len(self.memory) > self.config.batch_size * 5 :
            for _ in range(5) :
                experiences = self.memory.sample()
                self.update(experiences)

    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        preds = self.target_network(next_states, actions)
        q_targets = rewards + self.config.gamma * preds['Q_values'] * dones
        

        

    def reset(self):
        self.noise.reset()

    def soft_update(self):
        tau = self.config.tau_ddpg
        for targetp, netp in zip(self.target_network.parameters(), self.network.parameters()):
            self.target_network.data.copy_(tau * netp + (1 - tau) * targetp)
