import torch
import numpy as np 
import gym

from agent.PPOAgent import PPOAgent
from configs.config import Config


def main():

    config = Config()
    config.hidden_layer = 100
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.entropy_weight = 0.01

    env = gym.make('MountainCarContinuous-v0')
    max_steps = env.spec.timestep_limit
    print(max_steps)
    obs_size = np.shape(env.observation_space)[0]
    action_size = np.shape(env.action_space)[0]
    print(obs_size, action_size)
    agent = PPOAgent(config, obs_size, action_size, env)
    
    for i in range(10000):
        print('iter', i)
        states, actions, log_probs_old, returns, advantages = agent.sample()
        agent.ppo_update(states, actions, log_probs_old, returns, advantages)

    #print('returns', returns)
   
if __name__ == "__main__":
    main()   
