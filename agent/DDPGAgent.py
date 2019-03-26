import torch 
import numpy as np
from utils.noise import OrnsteinUhlenbeckProcess
from network.Networks import *
from .BaseAgent import BaseAgent
from network.Networks import DDPGActorCritic
from utils.memory import ReplayBuffer 

class DDPGAgent(BaseAgent):
    def __init__(self, config, state_size, action_size, hidden_size, env): 
        super(DDPGAgent, self).__init__()
        self.config = config
        self.network = ActorCriticDeterministic(state_size, action_size, config.hidden_layers)
        self.target_network = ActorCriticDeterministic(state_size, action_size, config.hidden_layers)
        self.target_network.load_state_dict(self.network.state_dict())
        self.memory = ReplayBuffer(action_size, buffer_size=1e6, batch_size=config.batch_size)
        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.network.critic.parameters())
        self.noise = OrnsteinUhlenbeckProcess(state_size, seed=445684)
        self.env = env
    
    def sample(self):
        state = self.env.reset()
        self.reset_noise()
        i = 1
        while True :
            action = self.network.forward_actor(state)
            if self.config.add_noise :
                action = action + self.noise.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(state, action, reward, next_state, done)
            i += 1
            if done :
                break
            if i > self.config.max_steps :
                break

    def step(self):
        if len(self.memory) > self.config.batch_size * 5 :
            for _ in range(5) :
                experiences = self.memory.sample()
                self.update(experiences)

    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        #====== Value loss ========
        value_criterion = nn.MSELoss()
        next_actions = self.target_network.forward_actor(next_states)
        target_value = self.target_network.forward_critic(next_states, next_actions)
        expected_value = rewards + self.config.gamma * target_value * dones
        value = self.network.forward_critic(states, actions)
        value_loss = value_criterion(expected_value, value)
        #====== Policy loss =======
        actions_ = self.network.forward_actor(states)
        policy_loss = -(self.network.forward_critic(states, actions_)).mean()
            
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
            self.target_network.data.copy_(tau * netp + (1 - tau) * targetp)
