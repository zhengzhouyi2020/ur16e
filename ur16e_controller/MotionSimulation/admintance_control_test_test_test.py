#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/20
# @Author : Zzy
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np

from ur16e_controller.MotionSimulation.control import UR16e_Controller
import ur16e_controller.utils.transform_utils as trans

plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体样式
plt.rcParams['font.size'] = 15  # 全局字体大小
plt.rcParams['axes.linewidth'] = 1


# 在坐标轴中画单个表
def data_plot(ax, x, y, xlabel, ylabel, title="", color='r', is_grid=False):
    ax.plot(x, y, color=color, linestyle='-', linewidth=0.8)
    ax.set_title(title, fontsize=15, )  # 设置标题
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


def smooth(a, WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # WSZ: 窗口口大小必须为奇数
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


class admittance_control:
    def __init__(self, trajectory, m=500, k=00, expect_force=-15):
        """
        尝试六自由度的导纳控制
        """
        # b = 2 * 1 * math.sqrt(m * k)
        b = 500
        self.m = m
        self.k = k
        self.b = b
        self.ctrl = UR16e_Controller()
        self.M = m * np.identity(3)
        self.B = b * np.identity(3)
        self.K = k * np.identity(3)
        self.exp_force = expect_force
        self.trajectory = np.array(trajectory)

    def move(self):
        length = len(self.trajectory)
        desire_vel = np.zeros([length, 3])
        desire_acc = np.zeros([length, 3])
        delta_d_p = 0
        # for i in range(len(self.trajectory) - 1):
        #     desire_vel[i + 1, :] = (self.trajectory[i + 1, :3] - self.trajectory[i, :3]) / 0.02
        #     desire_acc[i + 1, :] = (desire_vel[i + 1, :3] - desire_vel[i, :3]) / 0.02
        # desire_vel[i + 1, :] = 0
        # desire_acc[i + 1, :] = 0

        pre_p = 0
        pre_v = 0
        for i in range(len(self.trajectory) - 1):
            """
            还是不考虑姿态的变化了，太麻烦了
            """
            position = self.trajectory[i + 1, 0: 3]
            orientation = self.trajectory[i + 1, 3:7]

            delta_F_tool = np.array(self.getForce())[2] - self.exp_force  # 与期望力的误差

            # 计算导纳 ， 需要考虑一下用轨迹进行计算位置，而不是定点
            tran = np.identity(4)
            tran[:3, :3] = self.getRotateMatrix()
            tran[:3, 3] = self.getPosition()
            print(tran)
            print(position)
            x_d_e_ = np.dot(np.linalg.inv(tran), np.hstack([position, 1]))[2]  # Z方向需要运动的量

            print(x_d_e_)

            delta_dd_p = 1 / self.m * (delta_F_tool - self.b * pre_v - self.k * pre_p)
            delta_d_p = pre_v + delta_dd_p * 0.02
            delta_p = pre_p + delta_d_p * 0.02

            transform = trans.pose2mat(self.trajectory[i + 1, 0:7])
            contact = np.dot(transform, [0, 0, delta_p, 1])

            pose = np.array(
                [contact[0], contact[1], contact[2], orientation[0], orientation[1], orientation[2], orientation[3]])

            pre_v = delta_d_p
            pre_p = delta_p
            jnt = self.ctrl.ik(pose)
            self.ctrl.move_to_joint(jnt, total_time=0.02, isRecord=True)

    def getPosition(self):
        return np.array(self.ctrl.sim.data.get_site_xpos("ee_site") - self.ctrl.sim.data.get_body_xpos("base_link"))

    def getRotateMatrix(self):
        return np.array(self.ctrl.sim.data.get_site_xmat("ee_site"))

    def get66Matrix(self, matrix):
        length = len(matrix)
        left = np.row_stack((matrix, np.zeros((length, length))))
        right = np.row_stack((np.zeros((length, length)), matrix))
        result = np.hstack((left, right))
        return result

    def getPositionVel(self):
        return np.array(self.ctrl.sim.data.get_site_xvelp("ee_site"))

    def getOrientationVel(self):
        return np.array(self.ctrl.sim.data.get_site_xvelr("ee_site"))

    def getForce(self):
        return -np.array(self.ctrl.sim.data.sensordata)


def main():
    trajectory = np.loadtxt("../data/20220421_164243.csv", skiprows=1, delimiter=",")
    trajectory = trajectory[0:1000, ]

    controller = admittance_control(trajectory[100:1000])

    controller.ctrl.move_to_joint(trajectory[0, 7:13])
    controller.ctrl.move_to_joint(trajectory[100, 7:13])

    controller.move()
    l = controller.ctrl.force_list
    l = np.array(l)
    z_list = np.array(l[:, 2])
    z_list = smooth(z_list, 49)
    length = [i * 0.02 for i in range(len(z_list))]
    fig1 = plt.figure()
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.4, hspace=0.4)
    # plt.title("force",pad= 10,fontsize=20)
    ax1 = fig1.add_subplot(111)
    data_plot(ax1, x=length, y=z_list, xlabel="Time/s", ylabel="Force of z-axis/N", is_grid=True)
    np.savetxt("m200b500k500.csv", z_list)
    plt.show()
    # for i in range(len(trajectory)):
    #     controller.ctrl.move_to_joint(trajectory[i, 7:13],total_time=0.02)
    exit()


if __name__ == '__main__':
    main()
