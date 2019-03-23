import torch 
import numpy as np
from utils.noise import OrnsteinUhlenbeckProcess
from network.Networks import *
from .BaseAgent import BaseAgent
from network.Networks import ActorCriticContinous
from utils.memory import ReplayBuffer 

class DDPGAgent(BaseAgent):
    def __init__(self, config, state_size, action_size, hidden_size, env): 
        super(DDPGAgent, self).__init__()
        self.network = ActorCriticContinous(state_size, action_size, hidden_size)
        self.target_network = ActorCriticContinous(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.network.state_dict())
        self.memory = ReplayBuffer(action_size, buffer_size=1e4, batch_size=64)
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.noise = OrnsteinUhlenbeckProcess(state_size, seed=445684)
    
    def act(self):
        pass

    def step(self):
        pass

    def soft_update(self):
        pass
