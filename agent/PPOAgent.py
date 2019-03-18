import sys 
from .BaseAgent import *
from ..network.Networks import Actor, Critic
from ..utils.utils import compute_gae
from ..utils.storage import Storage

class PPOAgent(BaseAgent):

    def __init__(self, config, act_size, obs_size):
        BaseAgent.__init__(self, config)
        self.actor = Actor(obs_size, act_size, config.actor_layers)
        self.critic = Critic(obs_size, act_size, config.hidden_layers)
        self.config = config
        self.storage = Storage(config.rollout_length)
        
    def act(self, state):
        action = self.actor(state)
        return action

    def sampling(self):
        pass

    
    
        
    
