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


class admittance_control:
    def __init__(self, trajectory, m=10, k=0, expect_force=-15):
        """
        尝试六自由度的导纳控制
        """
        # b = 2 * 1 * math.sqrt(m * k)
        b = 200
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
        for i in range(len(self.trajectory) - 1):
            if i == 0:
                desire_vel[i + 1, :] = (self.trajectory[i + 1, :3] - self.trajectory[i, :3]) / 0.02
                desire_acc[i + 1, :] = (self.trajectory[i + 1, :3] - 2 * self.trajectory[i, :3] + self.trajectory[i,
                                                                                                  :3]) / (0.02 * 0.02)
            elif i == len(self.trajectory) - 1:
                desire_vel[i + 1, :] = (self.trajectory[i, :3] - self.trajectory[i - 1, :3]) / 0.02
                desire_acc[i + 1, :] = (self.trajectory[i, :3] - 2 * self.trajectory[i, :3] + self.trajectory[i - 1,
                                                                                              :3]) / (0.02 * 0.02)
            else:
                desire_vel[i + 1, :] = (self.trajectory[i + 1, :3] - self.trajectory[i - 1, :3]) / (0.02 * 2)
                desire_acc[i + 1, :] = (self.trajectory[i + 1, :3] - 2 * self.trajectory[i, :3] + self.trajectory[i - 1,
                                                                                                  :3]) / (0.02 * 0.02)

            # desire_vel[i + 1, :] = 0
            # desire_acc[i + 1, :] = 0

        pre_p = self.getPosition()[0:3]
        pre_v = self.getPositionVel()[0:3]
        for i in range(len(self.trajectory) - 1):
            """
            还是不考虑姿态的变化了，太麻烦了
            """
            cur_position = self.getPosition()  # 当前末端位姿
            cur_pvel = self.getPositionVel()  # 线速度

            position = self.trajectory[i + 1, 0: 3]
            orientation = self.trajectory[i + 1, 3:7]

            delta_F_tool = np.array(self.getForce())[2] - self.exp_force  # 与期望力的误差

            delta_F_tool = np.array([0, 0, delta_F_tool])

            rotateMatrix = self.getRotateMatrix()  # 获取旋转矩阵

            delta_F_base = np.dot(rotateMatrix, delta_F_tool)  # 计算基坐标系下的力

            delta_position = cur_position - position  # 计算与下一时刻位置偏差
            #
            delta_vel = cur_pvel - desire_vel[i + 1, :]

            # 计算导纳 ， 需要考虑一下用轨迹进行计算位置，而不是定点
            B = np.dot(self.B, delta_vel)
            K = np.dot(self.K, delta_position)

            # dd_p =desire_acc[i + 1, :] + np.dot(np.linalg.inv(self.M), (delta_F_base - B - K))  # 计算加速度
            # d_p = pre_v + dd_p * 0.02  # 加速度积分计算速度
            # position = pre_p + d_p * 0.02  # 速度积分计算位置
            # pose = np.hstack([position, orientation])

            delta_dd_p = np.dot(np.linalg.inv(self.M), (delta_F_base - B - K))
            delta_d_p = delta_vel + delta_dd_p * 0.02
            delta_p = delta_position + delta_d_p * 0.02
            #  position = position + delta_p
            transform = trans.pose2mat(self.trajectory[i + 1, 0:7])
            contact = np.dot(transform, [0, 0, delta_p[2], 1])

            pose = np.array(
                [contact[0], contact[1], contact[2], orientation[0], orientation[1], orientation[2], orientation[3]])

            # delta_temp = np.dot(rotateMatrix.transpose(), delta_p)
            # delta_temp[0], delta_temp[1] = 0, 0
            # delta_distance = np.dot(rotateMatrix, delta_temp)
            # position = position + delta_distance
            # pose = np.hstack([position, orientation])

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
    length = [i for i in range(len(z_list))]
    fig1 = plt.figure()
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.4, hspace=0.4)
    # plt.title("force",pad= 10,fontsize=20)
    ax1 = fig1.add_subplot(111)
    data_plot(ax1, x=length, y=z_list, xlabel="step", ylabel="Z Force", is_grid=True)
    plt.show()
    # for i in range(len(trajectory)):
    #     controller.ctrl.move_to_joint(trajectory[i, 7:13],total_time=0.02)
    exit()


if __name__ == '__main__':
    main()
