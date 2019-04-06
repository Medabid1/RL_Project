import gym 
import torch
import numpy as np

from configs.config import Config
from agent.DDPGAgent import DDPGAgent
from utils.utils import get_env_params
from utils.her import her_sampler

def main():
    env_names = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1','FetchSlide-v1']
    env = gym.make(env_names[3])
    #env.render()
    config = Config()
    config.render = False
    config.max_steps = env.spec.timestep_limit
    env_params = get_env_params(env)
    print(env_params)
    agent = DDPGAgent(config, env, env_params, her_sampler)
    agent.learn()
    #print('returns', returns)
   
if __name__ == "__main__":
    main()   
