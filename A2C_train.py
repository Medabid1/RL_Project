import sys
import torch
import gym
import numpy as np 
from agent.A2CAgent import A2CAgent
from configs.config import Config


def main():

    config = Config()
    config.actor_layers = [200,100]
    config.critic_layers = [200, 100]
    import gym
    env = gym.make('Humanoid-v2')
    max_steps = env.spec.timestep_limit
    obs_size = np.shape(env.observation_space)[0]
    action_size = np.shape(env.action_space)[0]
    agent = A2CAgent(config, obs_size, action_size)
    returns = []
    observations = []
    actions = []
    for i in range(config.rollout_lenght):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = 0
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            print('reward',r)
           
            totalr += r
            steps += 1
            if args.render:
               env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                 break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == "__main__":
    main()   

    


