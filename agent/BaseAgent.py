import torch 
import numpy as np 
import torch.multiprocessing as mp
from collections import deque
import sys


class BaseAgent:
    def __init__(self, config):
        self.config = config

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
    
    def load(self, filename):
        state_dict = torch.load(filename, map_location={lambda storage , loc : storage})
        self.network.load_state_dict(state_dict)
    