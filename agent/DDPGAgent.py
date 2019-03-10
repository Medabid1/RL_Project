import torch 
import numpy as np

from network.Networks import *
from .BaseAgent import BaseAgent

class DDPGAgent(BaseAgent):
    def __init__(self): 
        super(DDPGAgent, self).__init__()
