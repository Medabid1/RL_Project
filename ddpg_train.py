import gym 
import torch
import numpy as np

from configs.config import Config
from agent.DDPGAgent import DDPGAgent

def main():
    env = gym.make('FetchPickAndPlace-v1')
    env.render()
    config = Config()
    config.hidden_layers = [200,100]
    config.discount = 0.99
    config.batch_size = 64
    config.buffer_size = 1e6
    config.max_steps = env.spec.timestep_limit
    observation = env.reset()
    
    obs_size = np.shape(observation['observation'])[0]
    
    action_size = np.shape(env.action_space)[0]
    print(obs_size, action_size)
    agent = DDPGAgent(config=config, state_size=obs_size, action_size=action_size, env=env)
    
    for i in range(10000):
        print('iter', i)
        agent.sample()
        agent.step()
    #print('returns', returns)
   
if __name__ == "__main__":
    main()   
