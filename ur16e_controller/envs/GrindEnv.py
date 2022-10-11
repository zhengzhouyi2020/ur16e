#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/21
# @Author : Zzy


import sys

import os
import time
import math
import time
from typing import Tuple, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import mujoco_py
from gym.core import ActType, ObsType
from gym.envs.mujoco import mujoco_env
from gym import utils, spaces
from copy import deepcopy
import ur16e_controller.utils.transform_utils as trans

import traceback
from pathlib import Path
import copy

from ur16e_controller.MotionSimulation.control import UR16e_Controller
from ur16e_controller.controllers.joint_pos import JointPositionController
from ur16e_controller.utils.ur16e_kinematics import Kinematic


class GrindEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 trajectory,
                 expect_force=10,
                 file="/ur16e_grinding/ur16e.xml",
                 initial_point=None,
                 control_frequency=10,
                 Render=False
                 ):
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

        self.action_space = gym.spaces.Box(low=-0.01, high=0.01, shape=(1,), dtype=np.float32)

    def reset(self):
        self.index = 0
        if self.init_point is None:
            qpos = [0, -1.57, 1.57, -1.57, -1.57, 0.0]
            qvel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            qpos, qvel = self.init_point[:6], self.init_point[6:]
        qpos = np.array(qpos)
        qvel = np.array(qvel)
        self.set_state(qpos, qvel)
        self.sim.step()
        if self.Render:
            self.render()
        return self._get_obs()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        if not self.initialized:
            return self._get_obs(), 0, False, {}
        self.move(action)
        delta_force = abs(self.sim.data.sensordata[2] - self.expect_force)
        observation = self._get_obs()
        reward = self._get_reward(delta_force)
        information = self._get_info()
        done = self._get_done(delta_force)
        return observation, reward, done, information

    def _get_obs(self):
        """
        :return: 位置、速度、力传感器，或者其他关于位置偏差的数据
        """
        if not self.initialized:
            return np.zeros(13)
        else:
            return np.concatenate([self._get_pose(), self.sim.data.sensordata])

    def _get_pose(self):
        matrix = deepcopy(self.sim.data.get_site_xmat("ee_site"))
        quat = trans.mat2quat(matrix)
        position = self.sim.data.get_site_xpos("ee_site") - self.sim.data.get_body_xpos("base_link")
        return np.hstack([position, quat])

    def _get_info(self):
        info = dict()
        info['qpos'] = self.sim.data.qpos.copy()
        info['qvel'] = self.sim.data.qvel.copy()
        info['action'] = self.last_action
        return info

    def _get_reward(self, delta_force):
        reward = 0.0
        if delta_force > 15:
            reward = -500
        else:
            reward = 1 / (0.05 + 0.05 * delta_force ** 2)
        if self.index == len(self.trajectory):
            reward += 500
        return reward

    def _get_done(self, delta_force):
        if self.index == len(self.trajectory) or delta_force > 20:
            return True
        return False

    def move(self, action):
        """
        直接根据机器人的关节角度控制机器人,
        需要考虑控制的量是几个，可以是只考虑位置的改变，可以考虑姿态的改变
        :param action:
        :return:
        """
        pose = self.trajectory[self.index]
        position, orientation = pose[:3], pose[3:7]
        position = position + action  # 考虑位姿的转动
        quaternion = orientation
        if len(action) == 6:  # 仅考虑位置上的变动
            trans_mat = trans.quat2mat(orientation)
            quat_error = trans.axisangle2quat(action[3:])
            rotation_mat_error = trans.quat2mat(quat_error)
            trans_mat = np.dot(rotation_mat_error, trans_mat)
            quaternion = trans.mat2quat(trans_mat)
        pose = np.hstack([position, quaternion])

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
        pose = (orientation[:3], orientation[3:])
        Transform = trans.pose2mat(pose)
        jnt_init = self.sim.data.qpos
        jnt_out = self.ur16e_kinematics.IKine(Transform, jnt_init)
        return jnt_out
