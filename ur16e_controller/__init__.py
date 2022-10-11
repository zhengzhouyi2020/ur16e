#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/8
# @Author : Zzy

from gym.envs.registration import register
from ur16e_controller.version import VERSION as __version__

register(
    id='Grind-v0',
    entry_point="ur16e_controller.envs:GrindEnv",
)
register(
    id='Admittance-v0',
    entry_point="ur16e_controller.envs:AdmittanceEnv",
)
