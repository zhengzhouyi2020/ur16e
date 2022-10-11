#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/5/4
# @Author : Zzy
import argparse
import os.path

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from ur16e_controller.learn.PPO_continuous import Memory, PPO
from torch.utils.tensorboard import SummaryWriter

plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体样式
plt.rcParams['font.size'] = 15  # 全局字体大小
plt.rcParams['axes.linewidth'] = 1


def smooth(a, WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # WSZ: 窗口口大小必须为奇数
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


# 在坐标轴中画单个表
def data_plot(ax, x, y, xlabel, ylabel, title="", color='r', is_grid=False):
    ax.plot(x, y, color=color, linestyle='-', linewidth=0.8)
    ax.set_title(title, fontsize=9, )  # 设置标题
    ax.spines['right'].set_visible(False)  # 设置右侧坐标轴不可见
    ax.spines['top'].set_visible(False)  # 设置上坐标轴不可见
    ax.spines['top'].set_linewidth(0.6)  # 设置坐标轴线宽
    ax.spines['right'].set_linewidth(0.6)  # 设置坐标轴线宽
    ax.set_xlabel(xlabel, fontsize=15, labelpad=5)  # 设置横轴字体和距离坐标轴的距离
    ax.set_ylabel(ylabel, fontsize=15, labelpad=5)  # 设置纵轴字体和距离坐标轴的距离
    # ax.set_ylim(0,10000)  #设置y轴范围
    # ax.set_xlim(0, 10000)  # 设置x轴范围
    # 添加网格
    if is_grid:
        ax.grid(which='major', ls='--', alpha=.8, lw=.5)  # 是否设置网格，设置网格宽度和形状
    # 设置刻度坐标的朝向
    # ax.tick_params(which='major', x=5, width=1.5, direction='in', top='on', right="on")
    # ax.tick_params(which='minor', x=3, width=1, direction='in', top='on', right="on")


parser = argparse.ArgumentParser(description='PyTorch PPO for continuous controlling')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--env', type=str, default='Admittance-v0', help='continuous env')
parser.add_argument('--render', default=False, action='store_true', help='Render?')
parser.add_argument('--solved_reward', type=float, default=300000, help='stop training if avg_reward > solved_reward')
parser.add_argument('--print_interval', type=int, default=1, help='how many episodes to print the results out')
parser.add_argument('--save_interval', type=int, default=30, help='how many episodes to save a checkpoint')
parser.add_argument('--max_episodes', type=int, default=20)
parser.add_argument('--max_timesteps', type=int, default=800)
parser.add_argument('--update_timesteps', type=int, default=64, help='how many timesteps to update the policy')
parser.add_argument('--action_std', type=float, default=0.3,
                    help='constant std for action distribution (Multivariate Normal)')
parser.add_argument('--K_epochs', type=int, default=64, help='update the policy for how long time everytime')
parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipped')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--seed', type=int, default=123, help='random seed to use')
parser.add_argument('--ckpt_folder', default='./checkpoints', help='Location to save checkpoint models')
parser.add_argument('--tb', default=True, action='store_true', help='Use tensorboardX?')
parser.add_argument('--log_folder', default='./logs', help='Location to save logs')
parser.add_argument('--mode', default='train', help='choose train or test')
parser.add_argument('--restore', default=False, action='store_true', help='Restore and go on training?')
opt = parser.parse_args()

force_list = []
reward_list = []
m = []


def train(env_name, env, state_dim, action_dim, render, solved_reward,
          max_episodes, max_timesteps, update_timestep, action_std, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, restore, tb=False, print_interval=10, save_interval=100):
    ckpt = ckpt_folder + '/PPO_continuous_' + env_name + '.pth'
    if restore:
        print('Load checkpoint from {}'.format(ckpt))

    memory = Memory()

    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=restore, ckpt=ckpt)

    running_reward, avg_length, time_step = 0, 0, 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        t = 0
        force_list.clear()
        score = 0
        m.clear()
        for t in range(max_timesteps):
            time_step += 1

            # Run old policy
            action = ppo.select_action(state, memory)
            state, reward, done, info = env.step(action)
            m.append(action * 300 + 500)
            force_list.append(info['force'])
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += reward
            if render:
                env.render()

            if done:
                break
        avg_length += t
        reward_list.append(running_reward)
        if running_reward > (print_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
            print('Save a checkpoint!')
            break

        if i_episode % save_interval == 0:
            torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
            print('Save a checkpoint!')

        if i_episode % print_interval == 0:
            avg_length = int(avg_length / print_interval)
            running_reward = int((running_reward / print_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))

            if tb:
                writer.add_scalar('scalar/reward', running_reward, i_episode)
                writer.add_scalar('scalar/length', avg_length, i_episode)

            running_reward, avg_length = 0, 0


def tes(env_name, env, state_dim, action_dim, render, action_std, K_epochs, eps_clip, gamma, lr, betas, ckpt_folder,
        test_episodes):
    ckpt = ckpt_folder + '/PPO_continuous_' + env_name + '.pth'
    print('Load checkpoint from {}'.format(ckpt))

    memory = Memory()

    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=True, ckpt=ckpt)

    episode_reward, time_step = 0, 0
    avg_episode_reward, avg_length = 0, 0

    # test
    for i_episode in range(1, test_episodes + 1):
        state = env.reset()
        while True:
            time_step += 1

            # Run old policy
            action = ppo.select_action(state, memory)

            state, reward, done, _ = env.step(action)
            force_list.append(state[7:13])
            episode_reward += reward

            if render:
                env.render()

            if done:
                print('Episode {} \t Length: {} \t Reward: {}'.format(i_episode, time_step, episode_reward))
                avg_episode_reward += episode_reward
                avg_length += time_step
                memory.clear_memory()
                time_step, episode_reward = 0, 0
                break

    print('Test {} episodes DONE!'.format(test_episodes))
    print('Avg episode reward: {} | Avg length: {}'.format(avg_episode_reward / test_episodes,
                                                           avg_length / test_episodes))


if __name__ == '__main__':
    print(torch.__version__)
    trajectory = np.loadtxt("../data/20220415_151254.csv", delimiter=',', skiprows=1)
    initial_point = np.hstack([trajectory[0, 7:13], np.zeros(6)])

    if opt.tb:
        writer = SummaryWriter()

    if not os.path.exists(opt.ckpt_folder):
        os.mkdir(opt.ckpt_folder)

    print("Random Seed: {}".format(opt.seed))
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    env_name = opt.env
    env = gym.make(env_name, trajectory=trajectory, initial_point=initial_point, Render=False)
    env.seed(opt.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print('Environment: {}\nState Size: {}\nAction Size: {}\n'.format(env_name, state_dim, action_dim))

    if opt.mode == 'train':
        train(env_name, env, state_dim, action_dim,
              render=opt.render, solved_reward=opt.solved_reward,
              max_episodes=opt.max_episodes, max_timesteps=opt.max_timesteps, update_timestep=opt.update_timesteps,
              action_std=opt.action_std, K_epochs=opt.K_epochs, eps_clip=opt.eps_clip,
              gamma=opt.gamma, lr=opt.lr, betas=[0.9, 0.990], ckpt_folder=opt.ckpt_folder,
              restore=opt.restore, tb=opt.tb, print_interval=opt.print_interval, save_interval=opt.save_interval)
    elif opt.mode == 'test':
        tes(env_name, env, state_dim, action_dim,
            render=opt.render, action_std=opt.action_std, K_epochs=opt.K_epochs, eps_clip=opt.eps_clip,
            gamma=opt.gamma, lr=opt.lr, betas=[0.9, 0.990], ckpt_folder=opt.ckpt_folder, test_episodes=1)
    else:
        raise Exception("Wrong Mode!")
    if opt.tb:
        writer.close()

    l = np.array(force_list)
    l = smooth(l, 5)
    length = [i * 0.02 for i in range(len(force_list))]
    fig = plt.figure()
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.4, hspace=0.4)
    # plt.title("force",pad= 10,fontsize=20)
    ax1 = fig.add_subplot(111)
    data_plot(ax1, x=length, y=l[:], xlabel="step /s", ylabel="force_x  N", is_grid=True)

    # ax2 = fig.add_subplot(232)
    # data_plot(ax2, length, l[:, 1], "step", "force_y  N")
    #
    # ax3 = fig.add_subplot(233)
    # data_plot(ax3, length, l[:, 2], "step", "force_z  N")
    #
    # ax4 = fig.add_subplot(234)
    # data_plot(ax4, length, l[:, 3], "step", "force_x  N")
    #
    # ax5 = fig.add_subplot(235)
    # data_plot(ax5, length, l[:, 4], "step", "force_x  N")
    #
    # ax6 = fig.add_subplot(236)
    # data_plot(ax6, length, l[:, 5], "step", "force_x  N")

    reward_list = np.array(reward_list)
    length = [i for i in range(len(reward_list))]
    fig1 = plt.figure()
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.4, hspace=0.4)
    # plt.title("force",pad= 10,fontsize=20)
    ax2 = fig1.add_subplot(111)
    data_plot(ax2, x=length, y=reward_list, xlabel="step", ylabel="reward", is_grid=True)

    m = np.array(m)
    length = [i for i in range(len(m))]
    fig3 = plt.figure()
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.4, hspace=0.4)
    # plt.title("force",pad= 10,fontsize=20)
    ax3 = fig3.add_subplot(121)
    data_plot(ax3, x=length, y=m[:, 0], xlabel="step", ylabel="m", is_grid=True)
    ax3 = fig3.add_subplot(122)
    data_plot(ax3, x=length, y=m[:, 1], xlabel="step", ylabel="b", is_grid=True)
    print(np.mean(m[:, 0]))
    print(np.mean(m[:, 1]))

    plt.show()
