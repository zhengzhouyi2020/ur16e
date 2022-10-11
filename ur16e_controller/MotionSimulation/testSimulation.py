#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/2
# @Author : Zzy
import math
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from ur16e_controller.MotionSimulation.control import UR16e_Controller

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


def data_plot(ax, x, y, xlabel, ylabel, title="", color='r', is_grid=False):
    ax.plot(x, y, color=color, linestyle='-')
    ax.set_title(title)
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
    # ax.tick_params(which='major', length=5, width=1.5, direction='in', top='on', right="on")
    # ax.tick_params(which='minor', length=3, width=1, direction='in', top='on', right="on")


def main():
    ctrl = UR16e_Controller()  # 定义控制器

    ######################### 测试关节控制  ############################
    print(ctrl.sim.data.get_site_xpos("ee_site") - ctrl.sim.data.get_body_xpos("wrist_3_link"))

    target = [-1.57, -1.57, 1.57, -1.57, -0, 1.57]
    ctrl.move_to_joint(target, total_time=10, isRecord=False)
    print(ctrl.sim.data.qpos)

    ctrl.stay(10)

    ########################  测试点控  ##########################
    ctrl.move_to_point([-0.18056417, 0.47456404, 0.40209893, 3.14159265, 0, 0], 5, isRecord=False)
    ctrl.stay(10)
    ctrl.move_to_point([-0.18056417, 0.55, 0.01, 3.14159265, 0, 0], 5, isRecord=False)
    pose = [-0.4 + 0.22, 0.55, -0.01, 3.14159265, 0, 0]
    trajectory = []
    for i in range(1000):
        pose[0] += 0.8 / 1000
        trajectory.append(deepcopy(pose))
    pose[1] -= 0.1
    trajectory.append(deepcopy(pose))
    for i in range(1000):
        pose[0] -= 0.8 / 1000
        trajectory.append(deepcopy(pose))
    pose[1] -= 0.2

    ##########################  测试轨迹控制 ###########################
    ctrl.move_to_trajectory(trajectory, dt=0.02, isRecord=True)

    # ctrl.move_to_joint([0,0,0,0,0,0],total_time=10)
    # print(ctrl.sim.data.get_site_xpos("ee_site"))
    # print(ctrl.ur16e_kinematics.FKine(ctrl.sim.data.qpos))
    ########################### 轨迹信息保存 #####################
    # head = "px py pz qx qy qz qw " \
    #        "joint1 joint2 joint3 joint4 joint5 joint6 fx fy fz tx ty tz"
    # saveArray = np.hstack([ctrl.pose_list, ctrl.joint_position, ctrl.force_list])
    # ctrl.saveAsCSV(head, saveArray)

    ########################## 轨迹信息读取 #####################
    array = ctrl.loadFromCSV("../data/20220427_152752.csv")
    traject = array[:, 0:7]
    ctrl.move_to_point(pos=traject[0], total_time=5)  # 先运动到第一个点
    ctrl.move_to_trajectory(trajectory=traject)  # 复现轨迹
    l = ctrl.force_list
    l = np.array(l)
    length = [i * 0.02 for i in range(l.shape[0])]
    #
    # with plt.style.context(['ieee','grid']):
    #     plt.figure(23)
    #     plt.subplot(2, 3, 1)
    #     plt.plot(length, l[:, 0])
    #     plt.xlabel("step")
    #     plt.ylabel("force_x         N")
    #
    #     plt.subplot(2, 3, 2)
    #     plt.plot(length, l[:, 1])
    #     plt.xlabel("step")
    #     plt.ylabel("force_y         N")
    #
    #     plt.subplot(2, 3, 3)
    #     plt.plot(length, l[:, 2])
    #     plt.xlabel("step")
    #     plt.ylabel("force_z         N")
    #
    #     plt.subplot(2, 3, 4)
    #     plt.plot(length, l[:, 3])
    #     plt.xlabel("step")
    #     plt.ylabel("torque_x         N*m")
    #
    #     plt.subplot(2, 3, 5)
    #     plt.plot(length, l[:, 4])
    #     plt.title("ty")
    #     plt.xlabel("step")
    #     plt.ylabel("torque_y         N*m")
    #
    #     plt.subplot(2, 3, 6)
    #     plt.plot(length, l[:, 5])
    #     plt.title("tz")
    #     plt.xlabel("step")
    #     plt.ylabel("torque_z         N*m")
    #     plt.show()
    fig = plt.figure()
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.4, hspace=0.4)
    # plt.title("force",pad= 10,fontsize=20)
    ax1 = fig.add_subplot(231)
    data_plot(ax1, x=length, y=l[:, 0], xlabel="step /s", ylabel="Fx /N", is_grid=True)

    ax2 = fig.add_subplot(232)
    data_plot(ax2, length, l[:, 1], "step /s", "Fy /N", is_grid=True)

    ax3 = fig.add_subplot(233)
    data_plot(ax3, length, l[:, 2], "step /s", "Fz /N", is_grid=True)

    ax4 = fig.add_subplot(234)
    data_plot(ax4, length, l[:, 3], "step /s", "Tx /N*m", is_grid=True)

    ax5 = fig.add_subplot(235)
    data_plot(ax5, length, l[:, 4], "step /s", "Ty /N*m", is_grid=True)

    ax6 = fig.add_subplot(236)
    data_plot(ax6, length, l[:, 5], "step /s", "Tz /N*m", is_grid=True)

    fig1 = plt.figure()
    ax8 = fig1.add_subplot(111)
    data_plot(ax8, length, l[:, 5], "step /s", "joint1 /rad")

    plt.show()


if __name__ == '__main__':
    main()
