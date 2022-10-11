import time

import gym
import matplotlib.pyplot as plt
import ur16e_controller
import torch
from PPO import PPO
import numpy as np

position = []


def main():
    n_episodes = 100
    len_episode = 500
    batch = 32
    seed = 10

    trajectory = np.loadtxt("../data/20220415_151254.csv", delimiter=',', skiprows=1)
    initial_point = np.hstack([trajectory[0, 7:13], np.zeros(6)])

    env = gym.make("Grind-v0", trajectory=trajectory, initial_point=initial_point, Render=False)
    env.print_info()
    env.seed(seed)
    torch.manual_seed(seed)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    bound = env.action_space.high

    agent = PPO(n_states, n_actions, bound)
    all_ep_r = []
    for episode in range(n_episodes):
        ep_r = 0
        s = env.reset()
        position.clear()
        states, actions, rewards = [], [], []
        for t in range(len_episode):
            a = agent.choose_action(s)
            action = [0, 0, a.item()]
            time.sleep(0.002)
            s_, r, done, _ = env.step(action)
            position.append(s_[9])
            # env.render()
            ep_r += r
            states.append(s)
            actions.append(a)
            rewards.append((r + 8) / 8)  # 参考了网上的做法

            s = s_

            if (t + 1) % batch == 0 or t == len_episode - 1:  # N步更新
                targets = agent.discount_reward(rewards, s_)  # 奖励回溯
                agent.update(states, actions, targets)  # 进行actor和critic网络的更新
                states, actions, rewards = [], [], []
            if done:
                break
        print(episode)
        print(ep_r)
        # print('Episode {:03d} | Reward:{:.03f}'.format(episode, ep_r))

        # all_ep_r.append(ep_r)
        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.7 + ep_r * 0.3)  # 平滑
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.title("Reward")
    plt.show()


if __name__ == '__main__':
    main()
    length = len(position)
    position = np.array(position)
    print(position)
    plt.plot([i for i in range(length)], position[:])
    plt.xlabel("step")
    plt.ylabel("force")
    plt.title("Contact-Force")
    plt.show()
