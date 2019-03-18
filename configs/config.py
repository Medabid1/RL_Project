import argparse 
import torch 

class Config:
    
    def __init__(self):
        self.parser= argparse.ArgumentParser()
        self.discount = 0.99
        self.actor_layers = None
        self.critic_layers = None 
        self.rollout_length = 1000
        self.gradient_clip = None 
        self.use_gae = False 
        self.num_workers = None
        self.tau = 0.001
        self.batch_size = 256
        self.buffer_size = 100000
        self.optimization_rollouts = None
        
    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)