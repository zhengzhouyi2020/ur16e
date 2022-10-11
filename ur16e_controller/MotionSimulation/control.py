#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/2
# @Author : Zzy
import os
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import ur16e_controller.utils.transform_utils as trans

import numpy as np
from matplotlib import pyplot as plt

from ur16e_controller.controllers.joint_pos import JointPositionController
import mujoco_py

from ur16e_controller.utils.transform_utils import axisangle2quat, pose2mat
from ur16e_controller.utils.ur16e_kinematics import Kinematic


class UR16e_Controller(object):
    """
    Class for control of an robotic arm in MuJoCo.
    """

    def __init__(self, model=None, simulation=None, viewer=None, render=True, control_frequency=50,
                 isFilter=True, dim=6, windows=60) -> None:
        """
        :param model:
        :param simulation:
        :param viewer:
        :param render:
        :param control_frequency: 控制频率
        :param isFilter: 力是否滤波
        :param dim: 力的维度
        :param windows: 均值滤波窗口大小
        """
        super().__init__()
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent)
        if model is None:
            self.model = mujoco_py.load_model_from_path(path + '/ur16e_grinding/ur16e.xml')
        else:
            self.model = model
        if simulation is None:
            self.sim = mujoco_py.MjSim(self.model)
        else:
            self.sim = simulation
        if viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.viewer = viewer
        self.render = render
        self.control_frequency = control_frequency
        self.ur16e_kinematics = Kinematic()  # 机器人运动学类
        self.timeStep = self.sim.model.opt.timestep
        self.force_list = []  # 力列表
        self.joint_position = []  # 角位移列表
        self.velocity_list = []  # 速度列表
        self.accelerate_list = []  # 加速度列表
        self.pose_list = []
        self.marker_list = []
        self.filter = Average_Filter(dim, windows)
        self.isFilter = isFilter
        options = defaultdict()
        options["sim"] = self.sim
        options["actuator_range"] = (-150, 150)
        options["eef_name"] = "ee_site"
        joint_indexes = {
            "joints": [i for i in range(6)],
            "qpos": [i for i in range(6)],
            "qvel": [i for i in range(6)]
        }
        options["joint_indexes"] = joint_indexes
        options["output_max"] = 0.5
        options["output_min"] = -0.5
        options["kp"] = 100
        options["damping_ratio"] = 1
        self.robot_controller = JointPositionController(**options)

    def record(self):
        """
        :param isRecord: 是否记录数据
        :return:
        """
        if self.isFilter:
            self.filter.push(self.getForce())
            force = self.filter.average()
        else:
            force = self.getForce()
        self.joint_position.append(self.getQPos())
        self.force_list.append(force)
        self.filter.push(self.getForce())
        self.velocity_list.append(self.getQVel())
        self.accelerate_list.append(self.getQAcc())
        self.pose_list.append(self.getPose())
        self.marker_list.append(self.getPostion())

    def render_frame(self, position_list):
        # for i in range(len(position_list)):
        #     self.viewer.add_marker(pos = position_list[i],label = '',type = mujoco_py.generated.const.GEOM_SPHERE,size = [.01,0.01,0.01],rgba = np.ones(4))
        pass

    # def move_to_joint(self, target, total_time=2.0, isRecord=False):
    #     """
    #     :param target:  目标关节位移
    #     :param total_time: 运动总时间
    #     :param isRecord:  是否记录数据
    #     :return:
    #     """
    #     try:
    #         result = ''
    #         step = 0
    #         dt = 1 / self.control_frequency
    #         max_steps = int(total_time / dt)
    #         trajectory = TrajectoryJoint(target, self.sim.data.time, total_time, dt=self.timeStep,
    #                                      q_act=self.sim.data.qpos)
    #         while step < max_steps:
    #             qpos_ref = trajectory.next()
    #             self.robot_controller.set_goal(set_qpos=qpos_ref)
    #             torque = self.robot_controller.run_controller()
    #             self.sim.data.ctrl[:] = torque
    #             for i in range(int(dt / self.timeStep)):
    #                 trajectory.next()
    #                 self.sim.step()
    #             step = step + 1
    #             if isRecord:
    #                 self.record()
    #             if self.render:
    #                 self.viewer.render()
    #         return result
    #     except Exception as e:
    #         print(e)
    #         print('Could not move to requested joint target.')
    def move_to_joint(self, target, total_time=2.0, isRecord=False):
        """
        :param target:  目标关节位移
        :param total_time: 运动总时间
        :param isRecord:  是否记录数据
        :return:
        """
        try:
            result = ''
            max_steps = int(total_time / self.timeStep)
            for _ in range(max_steps):
                self.robot_controller.set_goal(set_qpos=target)
                torque = self.robot_controller.run_controller()
                self.sim.data.ctrl[:] = torque
                self.sim.step()
                if isRecord:
                    self.record()
                if self.render:
                    self.render_frame(self.marker_list)
                    self.viewer.render()
            return result
        except Exception as e:
            print(e)
            print('Could not move to requested joint target.')

    def stay(self, duration):
        """
        机器人停留在原地
        :param duration: 停止时间
        :param render: 渲染
        :return:
        """
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.move_to_joint(target=self.sim.data.qpos)
            elapsed = (time.time() - starting_time) * 1000

    def move_to_trajectory(self, trajectory, dt=0.02, isRecord=False):
        """
        轨迹控制
        :param trajectory: 运动轨迹
        :param dt: 运动间隔时间
        :param isRecord: 是否记录数据
        :return:
        """
        try:
            for i in range(len(trajectory)):
                # self.move_to_point(trajectory[i], dt, isRecord=isRecord)
                joint_angles = self.ik(trajectory[i])
                self.robot_controller.set_goal(set_qpos=joint_angles)
                torque = self.robot_controller.run_controller()
                self.sim.data.ctrl[:] = torque
                for j in range(int(dt / self.timeStep)):
                    self.sim.step()
                    if self.render:
                        self.render_frame(self.marker_list)
                        self.viewer.render()

                if self.render:
                    self.viewer.render()
                if isRecord:
                    self.record()
                print(self.sim.data.qpos)

        except Exception as e:
            print(e)
            print('Could not move along requested joint trajectory.')

    def move_to_point(self, pos, total_time=10.0, isRecord=False):
        """
        单点轨迹控制，可以通过四元数或者旋转矢量的方式进行控制
        :param pos: 位姿 （1×6 或 1×7）
        :param total_time: 运动时间
        :param isRecord: 是否记录数据
        :param control_frequency: 控制周期
        :return:
        """
        joint_angles = self.ik(pos)
        if joint_angles is not None:
            result = self.move_to_joint(target=joint_angles, total_time=total_time, isRecord=isRecord)
        else:
            result = 'No ,valid joint angles received, could not move EE to position.'
        return result

    def ik(self, orientation):
        """
        机器人运动学反解
        """
        if len(orientation) == 6:
            x = axisangle2quat(orientation[3:])
            pose = np.hstack([orientation[0:3], x])
        else:
            pose = orientation
        Transform = pose2mat(pose)
        jnt_init = self.sim.data.qpos
        jnt_out = self.ur16e_kinematics.IKine(Transform, jnt_init)
        return jnt_out

    def getForce(self):
        """获取传感器力数据"""
        sensor_list = self.sim.data.sensordata
        hex_force = -sensor_list[:6]
        return deepcopy(hex_force)

    def getPositionVel(self):
        """获取末端速度"""
        return deepcopy(self.sim.data.site_xvelp("ee_site"))

    def getOrientationVel(self):
        """获取末端速度"""
        return deepcopy(self.sim.data.site_xvelr("ee_site"))

    def getQAcc(self):
        """获取关节加速度"""
        return deepcopy(self.sim.data.qacc)

    def getQVel(self):
        """获取关节速度"""
        return deepcopy(self.sim.data.qvel)

    def getQPos(self):
        """获取关节速度"""
        return deepcopy(self.sim.data.qpos)

    def getPose(self):
        """获取位姿"""
        matrix = deepcopy(self.sim.data.get_site_xmat("ee_site"))
        quat = trans.mat2quat(matrix)
        position = self.sim.data.get_site_xpos("ee_site") - self.sim.data.get_body_xpos("base_link")
        return np.hstack([position, quat])

    def getPostion(self):
        position = self.sim.data.get_site_xpos("ee_site") - self.sim.data.get_body_xpos("base_link")
        return deepcopy(position)

    def saveAsCSV(self, header, array):
        """
        :param header: 文件中首行
        :param array: 保存的数组名
        """
        if not os.path.exists("../data"):
            os.makedirs("../data")
        uuid_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        tmp_file_name = '../data/%s.csv' % uuid_str
        np.savetxt(tmp_file_name, array, "%.6f", ',', header=header)

    def loadFromCSV(self, file):
        return np.loadtxt(file, delimiter=',', skiprows=1)  # 跳过第一行读取数据

    def move_single_point(self, pose, dt, isRecord=False):
        joint_angles = self.ik(pose)
        max_steps = dt / self.timeStep
        step = 0
        while step < max_steps:
            self.robot_controller.set_goal(set_qpos=joint_angles)
            torque = self.robot_controller.run_controller()
            self.sim.data.ctrl[:] = torque
            step = step + 1
            if isRecord:
                self.record()
            if self.render:
                self.viewer.render()


class TrajectoryGenerator:
    """
    关节空间三次插值
    """

    def __init__(self):
        self.extra_points = 0
        self.iterator = None

    def next(self):
        assert (self.iterator is not None)
        return next(self.iterator)


class TrajectoryJoint(TrajectoryGenerator):
    """
    关节空间的插值
    """

    def __init__(self, qd, initial_t, t_duration, dt=0.002, q_act=np.zeros((6,)), traj_profile=None):
        super(TrajectoryJoint, self).__init__()
        self.iterator = self._traj_override(qd, initial_t, t_duration, dt, q_act)

    def _traj_override(self, qd, initial_t=0, t_duration=1, dt=0.002, q_act=np.zeros((6,))):
        """
         Joint trajectory generation
        :param qd: joint space desired final point
        :param n: number of time steps
        :param initial_t: initial time
        :param t_duration: total time of travel
        :param dt: time step
        :param q_act: current joint position
        :param traj_profile: type of trajectory 'step', 'spline3', 'spline5', 'trapvel'
        :return: None
        """
        global q_ref
        n_timesteps = int(t_duration / dt)
        t_time = np.linspace(initial_t, t_duration + initial_t, n_timesteps)
        T = t_duration
        qvel0 = np.zeros((6,))  # V0
        qvelf = np.zeros((6,))  # Vf
        Q = np.array([q_act, qvel0, qd, qvelf])
        A_inv = np.linalg.inv(np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [1, T, T ** 2, T ** 3],
                                        [0, 1, 2 * T, 3 * T ** 2]]))
        coeffs = np.dot(A_inv, Q)
        for t in t_time:
            q_ref = np.dot(np.array([1, (t - initial_t), (t - initial_t) ** 2, (t - initial_t) ** 3]), coeffs)
            yield q_ref

        while True:
            self.extra_points = self.extra_points + 1
            yield q_ref


class Average_Filter:
    def __init__(self, dim, windows):
        self.dim = dim
        self.length = windows

        self._size = 0
        # Save pointer to end of buffer
        self.ptr = self.length - 1

        # Construct ring buffer
        self.buf = np.zeros((windows, dim))

    def push(self, value):
        # Increment pointer, then add value (also increment size if necessary)
        self.ptr = (self.ptr + 1) % self.length
        self.buf[self.ptr] = np.array(value)
        if self._size < self.length:
            self._size += 1

    def clear(self):
        self.buf = np.zeros((self.length, self.dim))
        self.ptr = self.length - 1
        self._size = 0

    def current(self):
        """
        Gets the most recent value pushed to the buffer

        Returns:
            float or np.array: Most recent value in buffer
        """
        return self.buf[self.ptr]

    def average(self):
        return np.mean(self.buf[: self._size], axis=0)
