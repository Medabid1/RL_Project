import sys 
import torch

from .BaseAgent import *
from ..network.Networks import ActorCritic
from ..utils.utils import compute_gae, random_sample
from ..utils.storage import Storage

class PPOAgent(BaseAgent):

    def __init__(self, config, act_size, obs_size, env):
        BaseAgent.__init__(self, config)
        self.network = ActorCritic(obs_size, act_size, 100)
        self.config = config
        self.env = env
        self.optimizer = torch.optim.Adam(self.network.parameters(), 3e-4, eps=1e-5) 

    def sample(self):
        env = self.env
        state = np.array(env.reset())
        self.storage = Storage(self.config.rollout_length)

        for _ in range(self.config.rollout_length):
            preditions = self.network(state)
            next_state, rewards, done, _ = env.step(preditions['action'])
            #reward = reward_normlizer(reward)
            
            self.storage.add(preditions)
            
            self.storage.add({'rewards': torch.Tensor(rewards).unsqueeze(-1),
                              'mask': torch.Tensor(1 - done).unsqueeze(-1),
                              'states': torch.Tensor(state)})
            
            state = np.array(next_state)
        
        self.storage.placeholder()
        advantages = torch.Tensor(np.zeros(1,1))
        returns = preditions['value'].detach()
        
        return compute_gae(advantages, returns, self.storage, self.config)

    def ppo_update(self, states, actions, log_probs, returns, advantages):

        for _ in range(self.config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), self.config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = torch.Tensor(batch_indices)
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
                                self.config.entroy_weight * prediction['entropy'].mean())

                value_loss = 0.5 * (sampled_returns - prediction['values']).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()            
        
    
    
        
    
