import gym 
import torch
import numpy as np

from configs.config import Config
from agent.DDPGAgent import DDPGAgent
from utils.utils import get_env_params
def main():
    env = gym.make('FetchSlide-v1')
    env.render()
    config = Config()
    config.hidden_layers = [200,100]
    config.discount = 0.99
    config.batch_size = 64
    config.buffer_size = 1e6
    config.max_steps = env.spec.timestep_limit
    config.use_her = True
    
    env_params = get_env_params(env)
    agent = DDPGAgent(config, obs_size, action_size, goal_size, env)
    
    for i in range(10000):
        print('iter', i)
        agent.sample()
        agent.step()
    #print('returns', returns)
   
if __name__ == "__main__":
    main()   
