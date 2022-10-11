#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/8
# @Author : Zzy

import gym
import matplotlib.pyplot as plt

import ur16e_controller  # 需要添加这一行，否则找不到环境注册的东西
import numpy as np
import time

trajectory = np.loadtxt("../data/20220414_robot_data_5_no_turn.csv", delimiter=',', skiprows=1)
trajectory = trajectory[:, 14:]
initial_point = np.hstack([trajectory[0, 7:13], np.zeros(6)])

env = gym.make("Grind-v0", trajectory=trajectory, initial_point=initial_point, Render=True)

# N_EPISODES = 100
# N_STEPS = 100
# env.print_info()
# plt.ion()
# f = plt.figure(1)
# for episode in range(N_EPISODES):
#     obs = env.reset()
#     plt.clf()
#     while True:
#         action = env.action_space.sample()
#         action = [0, 0, action[0]]
#         observation, reward, done, _ = env.step(action)
#         force = observation[8]
#         plt.plot(env.index, force,'.')
#         plt.pause(0.00001)
#         if done:
#             break
# plt.close(f)
# env.close()
# print("finished.")

N_EPISODES = 200
N_STEPS = 800
env.print_info()
position = []
for episode in range(N_EPISODES):
    obs = env.reset()
    for _ in range(N_STEPS):
        action = env.action_space.sample()
        action = [0, 0, action[0]]
        observation, reward, done, _ = env.step(action)
        position.append(observation[9])
length = len(position)
position = np.array(position)
print(position)
plt.plot([i for i in range(length)], position[:])
plt.xlabel("step")
plt.ylabel("force")
plt.title("Contact-Force")
plt.show()
