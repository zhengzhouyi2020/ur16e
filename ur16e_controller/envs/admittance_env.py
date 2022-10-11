#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/20
# @Author : Zzy
import math
import os

from typing import Tuple

import gym

import numpy as np

from gym.core import ActType, ObsType
from gym.envs.mujoco import mujoco_env
from gym import utils
from copy import deepcopy
import ur16e_controller.utils.transform_utils as trans

from pathlib import Path

from ur16e_controller.controllers.joint_pos import JointPositionController
from ur16e_controller.utils.ur16e_kinematics import Kinematic


class AdmittanceEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    为了减少参数的个数，设置 m = 0.1 k = 变量 ， b = 2 * sqrt(mk)
    """

    def __init__(self,
                 trajectory,
                 expect_force=-15.0,
                 file="/ur16e_grinding/ur16e.xml",
                 initial_point=None,
                 control_frequency=10,
                 Render=False
                 ):

        self.pre_p = 0
        self.pre_v = 0
        self.k = 10
        self.b = 500
        self.m = 200
        self.dF = 0.0
        self.ur16e_kinematics = Kinematic()
        self.current_observation = None
        self.initialized = False
        utils.EzPickle.__init__(self)
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent)
        full_path = path + file
        mujoco_env.MujocoEnv.__init__(self, full_path, control_frequency)
        self.control_frequency = control_frequency

        self.initialized = True
        self.TABLE_HEIGHT = 0.8845
        self.last_action = None
        self.timestep = self.sim.model.opt.timestep
        self.init_point = initial_point
        self.trajectory = trajectory  # 参考轨迹
        self.expect_force = expect_force  # 参考力
        self.index = 0

        options = dict()
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

        self.Render = Render

        self.action_space = gym.spaces.Box(np.array([-1, -1]).astype(np.float32),
                                           np.array([1, 1]).astype(np.float32))
        high = np.array([np.inf] * 2).astype(np.float32)
        self.observation_space = gym.spaces.Box(-high, high)

        self.control_step = self.timestep * control_frequency

    def reset_model(self):
        if self.init_point is None:
            qpos = [0, -1.57, 1.57, -1.57, -1.57, 0.0]
            qvel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            qpos, qvel = self.init_point[:6], self.init_point[6:]
        qpos = np.array(qpos)
        qvel = np.array(qvel)
        self.index = 0
        self.set_state(qpos, qvel)
        self.sim.forward()
        self.sim.step()
        if self.Render:
            self.render()
        self.pre_v = 0
        self.pre_p = 0
        return self._get_obs()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        if not self.initialized:
            return self._get_obs(), 0, False, {}
        self.move(action)
        self.dF = abs(-self.sim.data.sensordata[2] - self.expect_force)
        observation = self._get_obs()
        reward = self._get_reward(self.dF)
        information = self._get_info()
        done = self._get_done(self.dF)
        return observation, reward, done, information

    def _get_obs(self):
        """
        :return: 位置、速度、力传感器，或者其他关于位置偏差的数据
        """
        if not self.initialized:
            return np.zeros(2)
        else:

            return np.array([self.pre_p, self.dF], dtype=np.float32)

    def _get_pose(self):
        matrix = deepcopy(self.sim.data.get_site_xmat("ee_site"))
        quat = trans.mat2quat(matrix)
        position = self.sim.data.get_site_xpos("ee_site") - self.sim.data.get_body_xpos("base_link")
        return np.hstack([position, quat])

    def _get_info(self):
        info = dict()
        info['qpos'] = self.sim.data.qpos.copy()
        info['force'] = -self.sim.data.sensordata[2]
        info['action'] = self.last_action
        return info

    # TODO
    def _get_reward(self, delta_force):
        reward = 0.0
        if delta_force > 10:
            reward = -5
        else:
            reward = 1 / (0.05 + 0.05 * delta_force)
        if self.index == len(self.trajectory - 1):
            reward += 5
        return reward

    def _get_done(self, delta_force):
        if self.index == len(self.trajectory) - 1 or delta_force > 15:
            return True
        return False

    # TODO
    def move(self, action):
        """
        改变 M B K 的值， 即 M B K 的改变量，现在只考虑三个方向的导纳控制
        :param action: 已经归一化为 [-1,1], 希望能够在
        :return:
        """
        # k = 300. + action * 50.
        # damping_ratio = 1.
        # b = 2. * damping_ratio * np.sqrt(self.m * k)
        self.m = 500 + action[0] * 300
        self.k = 10
        damping_ratio = 0.7
        self.b = 500 + action[1] * 300

        pose = self.trajectory[self.index + 1]
        position, orientation = pose[:3], pose[3:7]

        delta_F_tool = np.array(self.getForce())[2] - self.expect_force  # 与期望力的误差

        # 计算导纳，计算期望轨迹的导纳控制

        delta_dd_p = 1 / self.m * (delta_F_tool - self.b * self.pre_v - self.k * self.pre_p)
        delta_d_p = self.pre_v + delta_dd_p * 0.02
        delta_p = self.pre_p + delta_d_p * 0.02

        transform = trans.pose2mat(self.trajectory[self.index + 1, 0:7])
        contact = np.dot(transform, [0, 0, delta_p, 1])

        pose = np.array(
            [contact[0], contact[1], contact[2], orientation[0], orientation[1], orientation[2], orientation[3]],
            dtype=object)

        self.pre_v = delta_d_p
        self.pre_p = delta_p

        joint_angles = self.ik(pose)
        for _ in range(self.frame_skip):
            self.robot_controller.set_goal(set_qpos=joint_angles)
            torque = self.robot_controller.run_controller()
            self.sim.data.ctrl[:] = torque
            self.sim.step()
            if self.Render:
                self.render()
        self.index = self.index + 1

    def close(self):
        mujoco_env.MujocoEnv.close(self)

    def print_info(self):
        print("Model timestep:", self.model.opt.timestep)
        print("Set number of frames skipped: ", self.frame_skip)
        print("dt = timestep * frame_skip: ", self.dt)
        print("Frames per second = 1/dt: ", self.metadata["video.frames_per_second"])
        print("ActionSpace: ", self.action_space)
        print("Observation space:", self.observation_space)

    def ik(self, orientation):
        if len(orientation) == 6:
            orientation[3:] = trans.axisangle2quat(orientation[3:])
        pose = orientation
        Transform = trans.pose2mat(pose)
        jnt_init = self.sim.data.qpos
        jnt_out = self.ur16e_kinematics.IKine(Transform, jnt_init)
        return jnt_out

    def getForce(self):
        return -np.array(self.sim.data.sensordata)
