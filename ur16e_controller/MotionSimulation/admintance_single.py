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
from ur16e_controller.utils.ur16e_kinematics import get_Jacobi

plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体样式
plt.rcParams['font.size'] = 15  # 全局字体大小
plt.rcParams['axes.linewidth'] = 1


class admittance_control:
    def __init__(self, trajectory, m=1, k=150, expect_force=-20):
        """
        尝试六自由度的导纳控制
        """
        b = 2 * np.sqrt(m * k)
        self.b = b
        self.m = m
        self.k = k
        self.ctrl = UR16e_Controller()
        self.exp_force = expect_force
        self.trajectory = np.array(trajectory)

    def move(self):
        length = len(self.trajectory)
        desire_vel = np.zeros([length])
        desire_acc = np.zeros([length])
        for i in range(len(self.trajectory) - 1):
            desire_vel[i + 1] = (self.trajectory[i + 1, 2] - self.trajectory[i, 2]) / 0.02
            desire_acc[i + 1] = (desire_vel[i + 1] - desire_vel[i]) / 0.02

            # desire_vel[i + 1] = 0
            # desire_acc[i + 1] = 0

        pre_p = self.getPosition()[2]
        pre_v = self.getPositionVel()[2]
        for i in range(len(self.trajectory) - 1):
            """
            还是不考虑姿态的变化了，太麻烦了
            """
            cur_position = self.getPosition()  # 当前末端位姿

            cur_pvel = self.getPositionVel()  # 线速度

            orientation = self.trajectory[i + 1, 3:7]

            delta_F_tool = np.array(self.getForce())[2] - self.exp_force  # 与期望力的误差

            rotateMatrix = self.getRotateMatrix()  # 获取旋转矩阵

            delta_F_base = np.dot(rotateMatrix, np.array([0, 0, delta_F_tool]))[2]  # 计算基坐标系下的力

            delta_p = cur_position[2] - self.trajectory[i + 1, 2]  # 计算与下一时刻位置偏差

            print("cur_pvel:", cur_pvel[2])
            print("pre_v:", pre_v)
            delta_vel = pre_v - desire_vel[i + 1]

            # 计算导纳 ， 需要考虑一下用轨迹进行计算位置，而不是定点
            B = self.b * delta_vel
            K = self.k * delta_p

            dd_p = desire_acc[i + 1, :] + np.dot(np.linalg.inv(self.m), (delta_F_base - B - K))  # 计算加速度
            d_p = cur_pvel + dd_p * 0.02  # 加速度积分计算速度
            position = cur_position + d_p * 0.02  # 速度积分计算位置
            pose = np.hstack([position, orientation])
            # dd_p = 1/self.m*(delta_F_base - B - K)  # 计算加速度
            # d_p = pre_v + dd_p * 0.02  # 加速度积分计算速度
            # position = pre_p + d_p * 0.02  # 速度积分计算位置
            # pre_v = d_p
            # pre_p = position
            # position = np.array([self.trajectory[1 + i, 0],self.trajectory[1 + i, 1],position])
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
        return -1 * np.array(self.ctrl.sim.data.sensordata)


def main():
    trajectory = np.loadtxt("../data/20220427_152752.csv", skiprows=1, delimiter=",")
    trajectory = trajectory
    controller = admittance_control(trajectory)
    controller.ctrl.move_to_joint([0, -1.57, 1.57, -1.57, -1.57, 0.0])
    controller.ctrl.move_to_joint(trajectory[0, 7:13])
    controller.move()
    l = controller.ctrl.force_list
    l = np.array(l)
    plt.plot([i for i in range(len(l))], l[:, 2])
    plt.show()
    # for i in range(len(trajectory)):
    #     controller.ctrl.move_to_joint(trajectory[i, 7:13],total_time=0.02)

    # 新建另一个图表
    fig1 = plt.figure(figsize=(16, 10))
    ax7 = fig1.add_subplot(111)  # 只有一个表
    ax7.plot([i for i in range(len(l))], l[:, 2], color='b', linestyle='-', linewidth=0.8, label="x-y")
    # 添加图例

    # 设置图例的位置，loc 后的数字一般取 1、2、3、4，依次表示右上、左上、左下和右下
    # prop 表示字体属性，我们选择预设的 edgecolor 表示图例边框颜色，w 为白色；framealpha 后的数字表示透明度
    # ncol=2 控制图例展示的列数；bbox_to_anchor=(x,y) 控制图例的位置。x 和 y 值为相对于坐标区的比例，依次为左右和上下
    ax7.set_title("Z contact force", fontsize=15, pad=15)  # 设置标题
    ax7.spines['right'].set_visible(False)  # 设置右侧坐标轴不可见
    ax7.spines['top'].set_visible(False)  # 设置上坐标轴不可见
    ax7.spines['top'].set_linewidth(0.6)  # 设置坐标轴线宽
    ax7.spines['right'].set_linewidth(0.6)  # 设置坐标轴线宽
    ax7.set_xlabel("step", fontsize=15, labelpad=5)  # 设置横轴字体和距离坐标轴的距离
    ax7.set_ylabel("z force  N", fontsize=15, labelpad=5)  # 设置纵轴字体和距离坐标轴的距离
    # ax.set_ylim(0,10000)  #设置y轴范围
    # ax.set_xlim(0, 10000)  # 设置x轴范围
    # 添加网格
    # ax7.grid(which='major', ls='--', alpha=.8, lw=.5)  # 是否设置网格，设置网格宽度和形状
    # 设置刻度坐标的朝向
    # ax.tick_params(which='major', x=5, width=1.5, direction='in', top='on', right="on")
    # ax.tick_params(which='minor', x=3, width=1, direction='in', top='on', right="on")

    plt.show()

    exit()


if __name__ == '__main__':
    main()
