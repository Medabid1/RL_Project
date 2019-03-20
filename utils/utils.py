import numpy as np
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)


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