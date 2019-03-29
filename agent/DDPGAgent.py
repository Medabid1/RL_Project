import torch
import torch.nn as nn 
import numpy as np
from utils.noise import OrnsteinUhlenbeckProcess
from .BaseAgent import BaseAgent
from network.Networks import ActorCriticDeterministic
from utils.memory import ReplayBuffer 
from utils.utils import to_tensor, to_numpy

class DDPGAgent(BaseAgent):
    def __init__(self, config, state_size, action_size, goal_size, env): 
        BaseAgent.__init__(self, config)
        self.config = config
        self.network = ActorCriticDeterministic(state_size, action_size, goal_size,
                                                config.hidden_layers, use_her=config.use_her)
        self.target_network = ActorCriticDeterministic(state_size, action_size, goal_size,
                                                    config.hidden_layers, use_her=config.use_her)
        self.target_network.load_state_dict(self.network.state_dict())
        self.memory = ReplayBuffer(action_size, buffer_size=int(config.buffer_size),
                                   batch_size=config.batch_size)
        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.network.critic.parameters())
        self.noise = OrnsteinUhlenbeckProcess(action_size, seed=445684)
        self.env = env
    
    def sample(self):
        observation = self.env.reset()
        goal = observation['desired_goal']
        state = observation['observation']
        self.reset_noise()
        i = 0
        totalreward = []
        while True :
            self.env.render()
            action = self.network.forward_actor(state, goal)
            if self.config.add_noise :
                action = action #+ to_tensor(self.noise.sample())
            print('recieving', self.env.step(to_numpy(action)))
            next_observation, reward, done, is_success = self.env.step(to_numpy(action))
            next_state = next_observation['observation']
            desired_goal = next_observation['desired_goal']
            
            totalreward.append(reward)
            self.memory.add(state, action, reward, next_state, done, goal)
            i += 1
            if done :
                break
            if i > self.config.max_steps :
                break
        print('Cummulative reward : {}, number of Steps : {}'.format(sum(totalreward), i))

    def step(self):
        if len(self.memory) > self.config.batch_size * 5 :
            for _ in range(5) :
                experiences = self.memory.sample()
                self.update(experiences)

    def update(self, experiences):
        states, actions, rewards, next_states, dones, goals = experiences
        #====== Value loss ========
        value_criterion = nn.MSELoss()
        next_actions = self.target_network.forward_actor(next_states, goals)
        target_value = self.target_network.forward_critic(next_states, next_actions, goals)
        expected_value = rewards + self.config.discount * target_value * dones
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
        #====== Soft update =========
        self.soft_update()

    def reset_noise(self):
        self.noise.reset()

    def soft_update(self):
        tau = self.config.tau_ddpg
        for targetp, netp in zip(self.target_network.parameters(), self.network.parameters()):
            targetp.data.copy_(tau * netp + (1 - tau) * targetp)
