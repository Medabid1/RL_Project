import sys
import torch
import gym
import numpy as np 
from agent.A2CAgent import A2CAgent
from configs.config import Config


def main():

    config = Config()
    config.actor_layers = [100,20]
    config.critic_layers = [100]
    env = gym.make('Humanoid-v2')
    max_steps = env.spec.timestep_limit
    print(max_steps)
    obs_size = np.shape(env.observation_space)[0]
    action_size = np.shape(env.action_space)[0]
    agent = A2CAgent(config, obs_size, action_size)
    returns = []
    observations = []
    actions = []
    for i in range(10000):
        print('iter', i)
        obs = env.reset()
        obs = np.array(obs[None, :], dtype=float)
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = agent.act(obs)
            observations.append(obs)
            actions.append(action)
            next_obs, r, done, _ = env.step(action)
            #print('reward',r)

            totalr += r
            steps += 1
            
            env.render()
            agent.step(obs, action, r, next_obs, done)
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                 break
        returns.append(totalr)
        
    #print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == "__main__":
    main()   

    


