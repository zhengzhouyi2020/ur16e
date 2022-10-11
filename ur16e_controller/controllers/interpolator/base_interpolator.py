#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/14
# @Author : Zzy
import abc


class Interpolator(object, metaclass=abc.ABCMeta):
    """
    General interpolator interface.
    """

    @abc.abstractmethod
    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        Returns:
            np.array: Next interpolated step
        """
        raise NotImplementedError
