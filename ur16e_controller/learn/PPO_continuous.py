#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/5/4
# @Author : Zzy
import torch
from torch import nn
from torch.distributions import Normal, MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:  # collected from old policy
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)  # (4, )

    def act(self, state, memory):  # state (1,24)
        action_mean = self.actor(state)  # (1,4)
        cov_mat = torch.diag(self.action_var).to(device)  # (4,4)
        # dist = Normal(action_mean, cov_mat)  # 这里当使用多维的时候，MultivariateNormal有修改
        dist = MultivariateNormal(action_mean, cov_mat)  # 多个action
        action = dist.sample()  # (1,4)
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):  # state (4000, 24); action (4000, 4)
        state_value = self.critic(state)  # (4000, 1)

        # to calculate action score(logprobs) and distribution entropy
        action_mean = self.actor(state)  # (4000,4)
        action_var = self.action_var.expand_as(action_mean)  # (4000,4)
        cov_mat = torch.diag_embed(action_var).to(device)  # (4000,4,4)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=False,
                 ckpt=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # current policy
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        if restore:
            pretained_model = torch.load(ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # old policy: initialize old policy with current policy's parameter
        self.old_policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  # flatten the state
        return self.old_policy.act(state, memory).cpu().numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimation of rewards
        # 还有一种对应的方法是TD法
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            # Evaluate old actions and values using current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Importance ratio: p/q
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages
            advantages = rewards - state_values.detach()

            # Actor loss using Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2)

            # Critic loss: critic loss - entropy
            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy

            # Total loss
            loss = actor_loss + critic_loss

            # Backward gradients
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights to old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())
