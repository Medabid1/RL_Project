import numpy as np
import torch.nn as nn
import torch 

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)

def to_tensor(x):
        if isinstance(x, torch.Tensor):
                return x.float()
        else: 
                return torch.from_numpy(x).float()

def to_numpy(x):
        if isinstance(x, np.ndarray):
                return x
        else :
                return x.detach().numpy()

def compute_gae(advantages, returns, storage, config):
    for i in reversed(range(config.rollout_length)):
            
            returns = storage.rewards[i] + (config.discount * storage.mask[i] * returns)

            td_error = (storage.rewards[i] + (config.discount *
                        storage.mask[i] * storage.values[i + 1]) - storage.values[i]) 
            advantages = td_error + (advantages * config.gae_tau
                                     * config.discount * storage.mask[i])

            storage.advantages[i] = advantages.detach()
            storage.returns[i] = returns.detach()
    
    states, actions, log_probs_old, returns, advantages = storage.cat(['states', 'actions', 
                                                        'log_prob', 'returns', 'advantages'])

    actions = actions.detach()
    log_probs_old = log_probs_old.detach()
    advantages = (advantages - advantages.mean()) / advantages.std()
    return states, actions, log_probs_old, returns, advantages

def reward_normliazer():
    pass
    
def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params