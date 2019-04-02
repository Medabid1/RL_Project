import torch
import torch.nn as nn 
import numpy as np

from .BaseAgent import BaseAgent
from utils.her import her_sampler
from utils.memory import ReplayBuffer 
from utils.utils import to_tensor, to_numpy
from utils.noise import OrnsteinUhlenbeckProcess
from network.Networks import ActorCriticDeterministic

class DDPGAgent(BaseAgent):

    def __init__(self, config, env, env_params, her): 
        BaseAgent.__init__(self, config)

        self.env = env
        self.config = config
        self.env_params = env_params
        
        self.network = ActorCriticDeterministic(env_params['obs'], env_params['action'],
                                                env_params['goal'],
                                                config.hidden_layers,
                                                 use_her=config.use_her)
        
        self.target_network = ActorCriticDeterministic(env_params['obs'], env_params['action'],
                                                       env_params['goal'], 
                                                       config.hidden_layers, 
                                                       use_her=config.use_her)

        self.target_network.load_state_dict(self.network.state_dict())
        
        self.her = her_sampler(config.replay_strategy, config.replay_k, 
                               env.compute_reward)

        self.replay_buffer = ReplayBuffer(env_params, 
                                          buffer_size=int(config.buffer_size),
                                          sample_func=self.her.sample_her_transitions)

        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.network.critic.parameters())
        self.noise = OrnsteinUhlenbeckProcess(env_params['action'], seed=445684)
    
    def learn(self):
        for epoch in range(self.config.n_epochs):
            self._sample()
            for  _ in range(self.config.batch_size):
                self._update()
            self._soft_update()

            if epoch % 10 == 0 :
                success_rate = self._eval_agent()
                print('Success rate in after {} epochs is {} over {} test runs'.format(epoch, 
                                                                                    success_rate,
                                                                                    self.config.test_rollouts))

                

    def _sample(self):
       
        obs_batch = []
        action_batch = []
        goals_batch = []
        achieved_goals_batch = []
        for _ in range(self.config.nb_rollouts): 
            actions_episode = []
            obs_episode = []
            goals_episode = []
            achieved_goals_episode = []
            observation = self.env.reset()
            goal = observation['desired_goal']
            obs = observation['observation']
            achieved_goal = observation['achieved_goal']
            self._reset_noise()
            i = 0
            while True :
                self.env.render()
                with torch.no_grad() : 
                    action = self.network.forward_actor(obs, goal)
                    if self.config.add_noise :
                        action = action + to_tensor(self.noise.sample())
                    action = to_numpy(action)
                #print('recieving', self.env.step(to_numpy(action)))
                print('action',action)
                print("obs",obs)
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

        self.replay_buffer.store_episode([np.array(obs_batch), 
                                          np.array(achieved_goals_batch), 
                                          np.array(goals_batch),
                                          np.array(action_batch)])

    
    def _update(self):
        experiences = self.replay_buffer.sample(self.config.batch_size)
        states = experiences['obs']
        actions = experiences['actions']
        next_states = experiences['next_obs']
        rewards = experiences['r']
        goals = experiences['g']
        next_goals = goals.copy()
        with torch.no_grad():
            next_actions = self.target_network.forward_actor(next_states, next_goals)
            target_value = self.target_network.forward_critic(next_states, next_actions, next_goals)
            expected_value = (to_tensor(rewards) + self.config.discount * target_value).detach()
            
            clip_return = 1 / (1 - self.config.discount)
            expected_value = torch.clamp(expected_value, -clip_return, self.config.clip_return or 0 )
    
        #====== Value loss ========
        value_criterion = nn.MSELoss()
        value = self.network.forward_critic(states, actions, goals)
        value_loss = value_criterion(expected_value, value)
        #====== Policy loss =======
        actions_ = self.network.forward_actor(states, goals)
        policy_loss = -(self.network.forward_critic(states, actions_, goals)).mean()    
        #====== Policy update =======
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        #====== Value update ========
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        

    def _reset_noise(self):
        self.noise.reset()

    def _soft_update(self):
        tau = self.config.tau_ddpg
        for targetp, netp in zip(self.target_network.parameters(), self.network.parameters()):
            targetp.data.copy_(tau * netp + (1 - tau) * targetp)

    def _eval_agent(self):
        total_success = []
        for _ in range(self.config.test_rollouts):
            local_success = []
            self.env.render()
            observation = self.env.reset()
            obs = observation['observation']
            goal = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    action = self.network.forward_actor(obs, goal)
                new_observation, _, _, info = self.env.step(to_numpy(action))
                obs = new_observation['observation']
                goal = new_observation['desired_goal']
                local_success.append(info['is_success'])
            
            total_success.append(local_success)
        
        total_success = np.array(total_success)
        
        return np.mean(total_success)