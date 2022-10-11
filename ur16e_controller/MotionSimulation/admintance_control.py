#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/20
# @Author : Zzy
import os
from collections import defaultdict
from pathlib import Path

import mujoco_py
import numpy as np

from ur16e_controller.MotionSimulation.control import UR16e_Controller
import ur16e_controller.utils.transform_utils as trans


class admittance_control:
    def __init__(self, trajectory, m=1, b=1, k=1, expect_force=10):
        """
        尝试六自由度的导纳控制
        """
        self.ctrl = UR16e_Controller()
        self.M = m * np.identity(6)
        self.B = b * np.identity(6)
        self.K = k * np.identity(6)
        self.exp_force = np.array([0, 0, expect_force, 0, 0, 0])
        self.trajectory = np.array(trajectory)

    def move(self):
        for i in range(len(self.trajectory)):
            """
            还是不考虑姿态的变化了，太麻烦了
            """
            cur_position = self.getPosition()  # 当前末端位姿
            cur_pvel = self.getPositionVel()  # 线速度
            cur_ovel = self.getOrientationVel()  # 角速度
            cur_vel = np.hstack([cur_pvel, cur_ovel])  # 当前速度
            delta_F_tool = np.array(self.getForce()) - self.exp_force  # 与期望力的误差
            rotateMatrix = self.getRotateMatrix()  # 获取旋转矩阵
            trans_matrix = self.get66Matrix(rotateMatrix)  # 获取力的变换矩阵
            delta_F_base = np.dot(trans_matrix, delta_F_tool)  # 计算基坐标系下的力
            delta_p = self.getPosition() - self.trajectory[i, 0:3]  # 计算位置偏差
            ref_orientation = trans.quat2mat(self.trajectory[i, 3:7])  # 期望姿态
            delta_orientation = np.dot(rotateMatrix, ref_orientation.T)  # 计算姿态偏差
            delta_orientation = trans.quat2axisangle(trans.mat2quat(delta_orientation))  # 将姿态偏差用轴角表示
            delta_pose = np.hstack([delta_p, delta_orientation])  # 位姿偏差 连接

            delta_vel = pre_vel - desire_vel

            # 计算导纳 ， 需要考虑一下用轨迹进行计算位置，而不是定点
            B = np.dot(self.B, delta_vel)
            K = np.dot(self.K, delta_pose)
            pre_dd_pose = np.dot(np.linalg.inv(self.M), (delta_F_base - B - K))  # 计算加速度

            d_pose = pre_vel + pre_dd_pose * self.ctrl.timeStep  # 加速度积分计算速度

            position = pre_pose[:3] + d_pose[:3] * self.ctrl.timeStep  # 速度积分计算位置

            orientation = np.dot(trans.quat2mat(trans.axisangle2quat(d_pose[3:7])), rotateMatrix)  # 姿态积分
            orientation = trans.mat2quat(orientation)
            pose = np.hstack([position, orientation])

            jnt = self.ctrl.ik(pose)
            self.ctrl.move_to_joint(jnt, total_time=0.02)

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
    trajectory = np.loadtxt("../data/20220415_151254.csv", skiprows=1, delimiter=",")
    trajectory = trajectory[:, 0:7]
    controller = admittance_control(trajectory)
    controller.ctrl.move_to_point(trajectory[0, :])
    controller.move()
    exit()


if __name__ == '__main__':
    main()
