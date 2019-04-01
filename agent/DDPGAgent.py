import torch
import torch.nn as nn 
import numpy as np
from utils.noise import OrnsteinUhlenbeckProcess
from .BaseAgent import BaseAgent
from network.Networks import ActorCriticDeterministic
from utils.memory import ReplayBuffer 
from utils.utils import to_tensor, to_numpy
from utils.her import her_sampler

class DDPGAgent(BaseAgent):
    def __init__(self, config, env, env_params, her): 
        BaseAgent.__init__(self, config)
        self.config = config
        self.her = her_sampler(config.replay_strategy, config.replay_k, env.compute_reward)
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
        
        self.replay_buffer = ReplayBuffer(env_params, 
                                          buffer_size=int(config.buffer_size),
                                          sample_func=self.her.sample_her_transitions)

        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.network.critic.parameters())
        self.noise = OrnsteinUhlenbeckProcess(env_params['action'], seed=445684)
        self.env = env
    
    def learn(self):
        for epoch in range(self.config.n_epochs):
            self._sample()
            for  _ in range(self.config.batch_size):
                self._update()
            self._soft_update()

            if epoch % 10 == 0 :
                self._eval_agent()
                

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
            obs = self.env.reset()
            goal = obs['desired_goal']
            state = obs['observation']
            achieved_goal = obs['achieved_goal']
            self.reset_noise()
            i = 0
            while True :
                #self.env.render()
                with torch.no_grad() : 
                    action = self.network.forward_actor(state, goal)
                    if self.config.add_noise :
                        action = action + to_tensor(self.noise.sample())
                #print('recieving', self.env.step(to_numpy(action)))
                obs, _, _, info = self.env.step(to_numpy(action))
                actions_episode.append(action)
                obs_episode.append(state)
                goals_episode.append(goal)
                achieved_goals_episode.append(achieved_goal)
            
                state = obs['observation']
                achieved_goal = obs['achieved_goal']
                i += 1
                if i >= self.env_params['max_timesteps'] :
                    break
            
            obs_batch.append(obs_episode)
            action_batch.append(actions_episode)
            achieved_goals_batch.append(achieved_goals_episode)
            goals_batch.append(goals_episode)

        self.replay_buffer.store_episode([obs_batch, 
                                          action_batch, 
                                          achieved_goals_batch,
                                          goals_batch])

    
    def _update(self):
        experiences = self.replay_buffer.sample(self.config.batch_size)
        states = experiences['obs']
        actions = experiences['actions']
        next_states = experiences['next_obs']
        rewards = experiences['r']
        goals = experiences['g']
        next_goals = experiences['next_g']
        actions, rewards, next_states, dones, goals = experiences
        #====== Value loss ========
        value_criterion = nn.MSELoss()
        next_actions = self.target_network.forward_actor(next_states, next_goals)
        target_value = self.target_network.forward_critic(next_states, next_actions, next_goals)
        expected_value = (rewards + self.config.discount * target_value * dones).detach()
        clip_return = 1 / (1 - self.config.discount)
        expected_value = torch.clamp(expected_value, -clip_return, 0)
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
        
        

    def reset_noise(self):
        self.noise.reset()

    def _soft_update(self):
        tau = self.config.tau_ddpg
        for targetp, netp in zip(self.target_network.parameters(), self.network.parameters()):
            targetp.data.copy_(tau * netp + (1 - tau) * targetp)

    def _eval_agent(self):
        #TODO do after each epoch, an evaluation and render here to win some time
        pass