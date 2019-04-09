import torch
import torch.nn as nn 
import numpy as np

from .BaseAgent import BaseAgent
from utils.her import her_sampler
from utils.memory import ReplayBuffer 
from utils.utils import to_tensor, to_numpy
from network.Networks import actor, critic
from utils.normalizers import normalizer

class DDPGAgent(BaseAgent):

    def __init__(self, config, env, env_params, her): 
        BaseAgent.__init__(self, config)

        self.env = env
        self.config = config
        self.env_params = env_params
        
        #self.network = ActorCriticDeterministic(env_params['obs'], env_params['action'],
        #                                        env_params['goal'],
        #                                       config.hidden_layers,
        #                                        use_her=config.use_her)
        self.actor = actor(env_params)
        self.target_actor = actor(env_params)
        self.target_actor.load_state_dict(self.actor.state_dict())
        #=============
        self.critic = critic(env_params)
        self.target_critic = critic(env_params)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # self.target_network = ActorCriticDeterministic(env_params['obs'], env_params['action'],
        #                                                env_params['goal'], 
        #                                                config.hidden_layers, 
        #                                                use_her=config.use_her)

        # self.target_network.load_state_dict(self.network.state_dict())
        
        self.her = her_sampler(config.replay_strategy, config.replay_k, 
                               env.compute_reward)

        self.replay_buffer = ReplayBuffer(env_params, 
                                          buffer_size=int(config.buffer_size),
                                          sample_func=self.her.sample_her_transitions)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.config.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.config.clip_range)
        self.model_path = '/home/mohamed/Desktop/Project/utils'
    def learn(self):
        for epoch in range(1,self.config.n_epochs+1):
            for _ in range(self.config.n_cycles):
                #for _ in range(2):
                episode = self._sample(epoch)
                self.replay_buffer.store_episode(episode)
                self._update_normalizer(episode)
                for  _ in range(self.config.n_batches):
                    self._update()
            
                self._soft_update()

            
            success_rate = self._eval_agent()
            
            print('Success rate in after {} epochs is {:.3f} over {} test runs'.format(epoch, 
                                                                                    success_rate,
                                                                                    self.config.test_rollouts))
        torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor.state_dict()],
                   self.model_path + '/model.pt')
                

    def _sample(self, epoch):
        obs_batch = []
        action_batch = []
        achieved_goals_batch = []
        goals_batch = []
        actions_episode = []
        obs_episode = []
        goals_episode = []
        achieved_goals_episode = []
        observation = self.env.reset()
        goal = observation['desired_goal']
        obs = observation['observation']
        achieved_goal = observation['achieved_goal']
        
        i = 0
        while True :
            if self.config.render :
                    self.env.render()
            with torch.no_grad() : 
                action = self.actor(obs, goal)
                
                if self.config.add_noise :
                    action = self._select_actions(action[0], 1/epoch)
                    
            new_obs, _, _, info = self.env.step(action)
            
            achieved_goal = new_obs['achieved_goal']
            obs_episode.append(obs.copy())
            obs = new_obs['observation']
            
            achieved_goals_episode.append(achieved_goal.copy())
            i += 1
            if i > self.env_params['max_timesteps'] :
                break
            actions_episode.append(action.copy())
            goals_episode.append(goal.copy())
        
            
        
        obs_batch.append(obs_episode)
        action_batch.append(actions_episode)
        achieved_goals_batch.append(achieved_goals_episode)
        goals_batch.append(goals_episode)
        
        episode = [np.array(obs_batch), np.array(achieved_goals_batch), np.array(goals_batch),
                                          np.array(action_batch)]
        
        # self.replay_buffer.store_episode([np.array(obs_batch), 
        #                                   np.array(achieved_goals_batch), 
        #                                   np.array(goals_batch),
        #                                   np.array(action_batch)])
        # self._update_normalizer([np.array(obs_batch), 
        #                                   np.array(achieved_goals_batch), 
        #                                   np.array(goals_batch),
        #                                   np.array(action_batch)])
        return episode

    def _update(self):
        experiences = self.replay_buffer.sample(self.config.batch_size)
        states, goals = self._preproc_og(experiences['obs'], experiences['g'])
        next_states, next_goals = self._preproc_og(experiences['next_obs'], experiences['g'])
        actions = experiences['actions']
        rewards = experiences['r']
        
        states = self.o_norm.normalize(states)
        goals = self.g_norm.normalize(goals)
        next_states = self.o_norm.normalize(next_states)
        next_goals = self.g_norm.normalize(next_goals)
        
        with torch.no_grad():
            next_actions = self.target_actor(next_states, next_goals)
            target_value = self.target_critic(next_states, next_actions[0], next_goals)
            expected_value = (to_tensor(rewards) + self.config.discount * target_value).detach()
            
            clip_return = 1 / (1 - self.config.discount)
            expected_value = torch.clamp(expected_value, -clip_return,  0 )
    
        #====== Value loss ========
        value_criterion = nn.MSELoss()
        value = self.critic(states, actions, goals)
        value_loss = value_criterion(expected_value, value)
        #====== Policy loss =======
        actions_ = self.actor(states, goals)
        policy_loss = -(self.critic(states, actions_[0], goals)).mean()    
        policy_loss += self.config.action_l2 * (actions_[0]).pow(2).mean()
        #====== Policy update =======
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        #====== Value update ========
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        


    def _soft_update(self):
        tau = self.config.tau_ddpg
        for targetp, netp in zip(self.target_critic.parameters(), self.critic.parameters()):
            targetp.data.copy_(tau * netp + (1 - tau) * targetp)

        for targetp, netp in zip(self.target_actor.parameters(), self.actor.parameters()):
            targetp.data.copy_(tau * netp + (1 - tau) * targetp)

    def _eval_agent(self):
        success_rate = 0
        for _ in range(self.config.test_rollouts):
            
            observation = self.env.reset()
            
            obs = observation['observation']
            goal = observation['desired_goal']
            obs , goal = self._preproc_inputs(obs, goal)
            for _ in range(self.env_params['max_timesteps']):
                if self.config.render :
                    self.env.render()
                with torch.no_grad():
                    action = self.actor(obs, goal)
                new_obs, _, _, info = self.env.step(to_numpy(action[0]))
                obs, goal = self._preproc_inputs(new_obs['observation'], new_obs['desired_goal'])
            success_rate += info['is_success']  
                
            
        return success_rate/self.config.test_rollouts

    def _select_actions(self, pi, eps):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
            
        action += self.config.eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1,eps,1) * (random_actions - action)
        return action

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'next_obs': mb_obs_next,
                       'next_ag': mb_ag_next,
                       }
        transitions = self.her.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
    
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return obs_norm, g_norm
    
    def _preproc_og(self, o, g):
        o = np.clip(o, -self.config.clip_obs, self.config.clip_obs)
        g = np.clip(g, -self.config.clip_obs, self.config.clip_obs)
        return o, g