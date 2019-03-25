import sys 
import torch
import numpy as np

from .BaseAgent import BaseAgent
from network.Networks import ActorCriticContinous
from utils.utils import compute_gae, random_sample
from utils.storage import Storage

class PPOAgent(BaseAgent):

    def __init__(self, config, obs_size, act_size, env):
        BaseAgent.__init__(self, config)
        self.network = ActorCriticContinous(obs_size, act_size, config.hidden_layer)
        self.config = config
        self.env = env
        self.optimizer = torch.optim.Adam(self.network.parameters(), 3e-4, eps=1e-5) 

    def sample(self):
        env = self.env
        state = np.array(env.reset())
        self.storage = Storage(self.config.rollout_length)
        cumm_return = 0

        for _ in range(self.config.rollout_length):
            self.env.render()
            preditions = self.network(state)
            next_state, rewards, done, _ = env.step(preditions['actions'])
            #reward = reward_normlizer(reward)
            self.storage.add(preditions)
            cumm_return += rewards    
            self.storage.add({'rewards': torch.from_numpy(np.asarray(rewards)).unsqueeze(-1).float(),
                              'mask': torch.from_numpy(np.asarray(1 - done)).unsqueeze(-1).float(),
                              'states': torch.from_numpy(state).float().unsqueeze(0)})
            # if done : 
            #     state = np.array(env.reset())
            # else : 
            state = np.array(next_state)

        print('cummulative return', cumm_return)
        print('avg return', cumm_return / self.config.rollout_length)
        
        prediction = self.network(state)
        self.storage.add(prediction)
        self.storage.placeholder()
        advantages = torch.Tensor(np.zeros((1)))
        returns = preditions['values'].detach()
        
        return compute_gae(advantages, returns, self.storage, self.config)

    def ppo_update(self, states, actions, log_probs, returns, advantages):

        for _ in range(self.config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), self.config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = torch.Tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs = log_probs[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_prob'] - sampled_log_probs).exp()

                obj = ratio * sampled_advantages
                obj_cliped = ratio.clamp(1.0 - self.config.ppo_clip,
                                         1.0 + self.config.ppo_clip) * sampled_advantages

                policy_loss = -(torch.min(obj, obj_cliped).mean() + 
                                self.config.entropy_weight * prediction['entropy'].mean())

                value_loss = 0.5 * (sampled_returns - prediction['values']).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()            
        
        print('policy loss', policy_loss)
        print('value loss', value_loss)
    
        
    
