#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/2
# @Author : Zzy
import math
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from ur16e_controller.MotionSimulation.control import UR16e_Controller

trajectory = np.loadtxt("../data/20220414_robot_data_3_no_turn.csv", delimiter=',', skiprows=1)
trajectory = trajectory[:, 13:]


def main():
    ctrl = UR16e_Controller()  # 定义控制器

    point = trajectory[0, 6:12]
    point[4] += 0.25
    ctrl.move_to_joint(point, total_time=3, isRecord=True)
    ctrl.stay(10000)
    for i in range(1, len(trajectory)):
        trajectory[i, 2] -= 0.06
    ctrl.move_to_trajectory(trajectory[:, 0:6], dt=0.02, isRecord=True)

    ##########################  测试轨迹控制 ##########################


if __name__ == '__main__':
    main()
