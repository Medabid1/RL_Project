import gym 
import torch
import numpy as np

from configs.config import Config
from agent.DDPGAgent import DDPGAgent
from utils.utils import get_env_params
from utils.her import her_sampler

def main():
    env = gym.make('FetchSlide-v1')
    config = Config()
    config.hidden_layers = [256, 256]
    config.discount = 0.99
    config.buffer_size = 1e6
    config.use_her = True
    config.max_steps = env.spec.timestep_limit
    env_params = get_env_params(env)
    agent = DDPGAgent(config, env, env_params, her_sampler)
    agent.learn()
    #print('returns', returns)
   
if __name__ == "__main__":
    main()   
