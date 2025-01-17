import torch 
import numpy as np 
import torch.multiprocessing as mp
from collections import deque
import sys


class BaseAgent:
    
    def __init__(self, config):
        self.config = config
        
    def save(self, filename, network):
        torch.save(network.state_dict(), filename)
    
    def load(self, network, filename):
        state_dict = torch.load(filename, map_location={lambda storage , loc : storage})
        network.load_state_dict(state_dict)
    
    def init_weights(self,m):
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.1)