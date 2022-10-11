#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/3
# @Author : Zzy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNet(nn.Module):
    def __init__(self, n_states, action_bound, fc1_dims=128, fc2_dims=128):
        super(ActorNet, self).__init__()
        self.n_states = n_states
        self.action_bound = torch.tensor(action_bound)

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU()
        )

        self.mu_out = nn.Linear(fc2_dims, 1)
        self.sigma_out = nn.Linear(fc2_dims, 1)

    def forward(self, x):
        x = torch.relu(self.layer(x))
        mu = torch.tanh(self.mu_out(x)) * self.action_bound
        sigma = F.softplus(self.sigma_out(x))
        return mu, sigma


class CriticNet(nn.Module):
    def __init__(self, n_states, fc1_dims=128, fc2_dims=128):
        super(CriticNet, self).__init__()
        self.n_states = n_states

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

    def forward(self, x):
        value = self.layer(x)
        return value


class PPO(nn.Module):
    def __init__(self, n_states, n_actions, action_bound, lr=0.0001, gamma=0.7, epsilon=0.2, a_update_steps=10,
                 c_update_steps=10):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.lr = lr  # 共用一个lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.a_update_steps = a_update_steps
        self.c_update_steps = c_update_steps

        self._build()

    def _build(self):
        self.actor_model = ActorNet(self.n_states, self.action_bound)
        self.actor_old_model = ActorNet(self.n_states, self.action_bound)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=self.lr)

        self.critic_model = CriticNet(self.n_states)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=self.lr)

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        mu, sigma = self.actor_model(s)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()

        return np.clip(action, -self.action_bound, self.action_bound)

    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_)
        target = self.critic_model(s_).detach()  # torch.Size([1])
        target_list = []
        for r in rewards[::-1]:
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_list = torch.cat(target_list)  # torch.Size([batch])

        return target_list

    def actor_learn(self, states, actions, advantage):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).reshape(-1, 1)

        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)

        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))
        surr = ratio * advantage.reshape(-1, 1)  # torch.Size([batch, 1])
        loss = -torch.mean(
            torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)

        loss_func = nn.MSELoss()
        loss = loss_func(v, targets)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def cal_adv(self, states, targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states)  # torch.Size([batch, 1])
        advantage = targets - v.reshape(1, -1).squeeze(0)
        return advantage.detach()  # torch.Size([batch])

    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())  # 首先更新旧模型
        advantage = self.cal_adv(states, targets)

        for i in range(self.a_update_steps):  # 更新多次
            self.actor_learn(states, actions, advantage)

        for i in range(self.c_update_steps):  # 更新多次
            self.critic_learn(states, targets)

    def save(self, episod):
        torch.save(self.critic_model.state_dict(), "./model/ppo_critic{}.pth".format(episode))
        torch.save(self.actor_model.state_dict(), "./model/ppo_actor{}.pth".format(episode))

    def load(self, episode):
        self.critic_model.load_state_dict(torch.load("./model/ppo_critic{}.pth".format(episode)))
        self.actor_model.load_state_dict(torch.load("./model/ppo_actor{}.pth".format(episode)))
