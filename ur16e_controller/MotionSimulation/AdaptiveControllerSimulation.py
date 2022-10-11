#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/2
# @Author : Zzy
import math

"""
@author:wy
referrence:
2011 IEEE T-RO Human-like adaptation of force and impedance in stable and unstable interactions;
2018 IEEE T-RO Force, impedance, and trajectory learning for contact tooling and haptic identification;
2021 IEEE ASME A unified parametric representation for robotic compliant skills with adaptation of impedance and force;

r2euler: 原文链接：https: // blog.csdn.net / DrGene / article / details / 121084553
"""
import sys

sys.path.append(r'/home/wendy/Code/pyrobolearn-master/pyrobolearn/models/dmp')

# from copy import deepcopy
from matplotlib import pyplot as plt
from ur16e_controller.MotionSimulation.control import UR16e_Controller
from ur16e_controller.controllers.joint_pos import JointPositionController
import numpy as np
# import pandas as pd
# import sympy
# import sympybotics
# import cv2
# from scipy.linalg import pinv
# import robot_danamic_params as rdp
# import ur_kinematics
from ur_kinematics import Kinematic
import jacobi
import math
from ur16e_controller.utils.transform_utils import mat2quat


class ImpedanceForceController_Cartesian:
    def __init__(self, gamma=np.eye(6), Ks=0. * np.eye(6), Kd=0.0001 * np.eye(6), FF=None,
                 Qs=0.06 * np.eye(6), Qd=0.03 * np.eye(6), Qf=5. * np.eye(6), Qr=0.0002 * np.eye(6), beta=0.00001):
        self.gamma = gamma
        self.Ks = Ks
        self.Kd = Kd
        self.FF = FF
        self.Qs = Qs
        self.Qd = Qd
        self.Qf = Qf
        self.Qr = Qr
        self.beta = beta

    def compute_sliding_error(self, x, xr, dt, alpha=30):
        e = np.zeros(6)
        # R1 = cv2.Rodrigues(xr[3:])[0]
        # R2 = cv2.Rodrigues(x[3:])[0]
        R1 = eulerAngles2rotationMat(xr[3:])
        R2 = eulerAngles2rotationMat(x[3:])

        dR = R2.dot(np.linalg.pinv(R1))

        # e[3:] = np.transpose(cv2.Rodrigues(dR)[0][0:3, 0])
        e[3:] = rotationMatrixToEulerAngles(dR)
        e[0:3] = x[0:3] - xr[0:3]

        # e = x - xr
        de = e / dt
        return de + alpha * e

    def compute_deta_xr(self, dzeta, dKs, dKd, xr):
        deta_xr = np.linalg.pinv(self.Ks).dot(dzeta - dKs * xr - dKd * xr)
        return deta_xr

    def compute_deta_zeta(self, Fd, xr, dxr):
        return self.Qr.dot(Fd - self.FF - (self.Ks.dot(xr) + self.Kd.dot(dxr)))

    def compute_dKs(self, x, xr, dt):
        epsilon = self.compute_sliding_error(x, xr, dt)
        return self.Qs * (epsilon * np.transpose(x) - self.beta * self.Ks)

    def compute_dKd(self, dx, x, xr, dt):
        epsilon = self.compute_sliding_error(x, xr, dt)
        return self.Qd.dot(epsilon.dot(np.transpose(dx)) - self.beta * self.Kd)

    def compute_dF(self, x, xr, dxr, Fd, dt):
        epsilon = self.compute_sliding_error(x, xr, dt)
        dzeta = self.compute_deta_zeta(Fd, xr, dxr)
        return self.Qf.dot(epsilon - self.beta * self.FF + np.transpose(self.Qr).dot(dzeta))

    def update_parameter(self, x, xr, dx, dxr, Fd, dt):
        self.Ks = self.Ks + self.compute_dKs(x, xr, dt)
        self.Kd = self.Kd + self.compute_dKd(dx, x, xr, dt)
        self.FF = self.FF + self.compute_dF(x, xr, dxr, Fd, dt)


class ImpedanceForceController_Joint:
    def __init__(self, gamma=2. * np.eye(6), Ks=0.5 * np.eye(6), Kd=20. * np.eye(6), tau=np.zeros((1, 6)),
                 Qs=50. * np.eye(6), Qd=0.03 * np.eye(6), Qf=3.6 * np.eye(6), Qr=0.02 * np.eye(6),
                 L=0.02 * np.eye(6), beta=0.00001):
        self.gamma = gamma
        self.Ks = Ks
        self.Kd = Kd
        self.tau = tau
        self.Qs = Qs
        self.Qd = Qd
        self.Qf = Qf
        self.Qr = Qr
        self.L = L
        self.beta = beta

    def compute_sliding_error(self, q, qr, dq, dqr, alpha=10.):
        e = q - qr
        de = dq - dqr
        return e, de, de + alpha * e

    def compute_dKs(self, q, qr, dq, dqr):
        e, de, epsilon = self.compute_sliding_error(q, qr, dq, dqr)
        return self.Qs.dot(epsilon * np.transpose(e) - self.beta * self.Ks)

    def compute_dKd(self, q, qr, dq, dqr):
        e, de, epsilon = self.compute_sliding_error(q, qr, dq, dqr)
        return self.Qd.dot(epsilon.dot(np.transpose(de)) - self.beta * self.Kd)

    def compute_dtau(self, q, qr, dq, dqr, tau_d):
        e, de, epsilon = self.compute_sliding_error(q, qr, dq, dqr)
        deta_sigma = np.linalg.pinv(np.transpose(self.L)).dot(self.Qr.dot(
            tau_d.reshape(6, 1) - self.tau.reshape(6, 1) - (self.Ks.dot(e) + self.Kd.dot(de)).reshape(6, 1))).reshape(6,
                                                                                                                      1)
        return self.Qf.dot(
            epsilon.reshape(6, 1) - (self.beta * self.tau).reshape(6, 1) + (np.transpose(self.Qr).dot(deta_sigma)))

    def update_parameter(self, q, qr, dq, dqr, tau_d=[0, 0, 0, 0, 0, 0]):
        self.Ks = self.Ks + self.compute_dKs(q, qr, dq, dqr)
        self.Kd = self.Kd + self.compute_dKd(q, qr, dq, dqr)
        self.tau = self.tau + self.compute_dtau(q, qr, dq, dqr, tau_d).reshape(1, 6)

    def update_Kd(self, q, qr, dq, dqr):
        self.Kd = self.Kd + self.compute_dKd(q, qr, dq, dqr)


class Pid():

    def __init__(self, exp_fz, kp, ki, kd, dt):
        self.KP = kp
        self.KI = ki
        self.KD = kd
        self.exp_fz = exp_fz
        self.dt = dt
        self.now_fz = 0
        self.sum_err = 0
        self.now_err = 0
        self.last_err = 0

    def pid_force_position(self, exp_fz=None, now_fz=None, ):
        self.exp_fz = exp_fz
        self.now_fz = now_fz
        self.last_err = self.now_err
        self.now_err = self.now_fz - self.exp_fz
        self.sum_err += self.now_err
        # 这一块是严格按照公式来写的
        Xe = self.KP * (self.now_err) \
             + self.KI * self.sum_err * self.dt
        return Xe


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAngles2rotationMat(theta, format='rad'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def main():
    # Initialize desired trajectory data 20211215_robot_data_10_no_turn.csv
    data = np.loadtxt(open(
        "/home/wendy/Code/pyrobolearn-master/pyrobolearn/models/dmp/Experimental_data/20220414_robot_data_3_no_turn.csv",
        "rb"), delimiter=",", skiprows=1)
    t = data[1000:1500, 0] - data[1000, 0]
    ft = data[1000:1500, 7:13]
    qr = data[1000:1500, 19:]

    # torque in joint space
    tau_j = np.zeros(np.shape(qr))
    for i in range(np.shape(t)[0]):
        J = jacobi.get_Jacobi(qr[i, :])
        tau_j[i, :] = np.transpose(J).dot(ft[i, :])

    # desire position
    dqr = np.zeros(np.shape(qr))
    ddqr = np.zeros(np.shape(qr))
    dt = np.zeros(np.shape(t))

    z = np.zeros((np.shape(t)[0], 3))
    dz = np.zeros((np.shape(t)[0], 3))
    ddz = np.zeros((np.shape(t)[0], 3))

    xr = np.zeros((np.shape(t)[0], 7))
    dxr = np.zeros((np.shape(t)[0], 6))
    ddxr = np.zeros((np.shape(t)[0], 6))

    # actual position
    q = np.zeros(np.shape(qr))
    dq = np.zeros(np.shape(qr))
    ddq = np.zeros(np.shape(qr))

    x = np.zeros((np.shape(t)[0], 7))
    dx = np.zeros((np.shape(t)[0], 6))
    ddx = np.zeros((np.shape(t)[0], 6))

    v = np.zeros((np.shape(t)[0], 6))
    w = np.zeros((np.shape(t)[0], 6))

    Ks = np.zeros((np.shape(t)[0], 6))
    Kd = np.zeros((np.shape(t)[0], 6))
    tau = np.zeros((np.shape(t)[0], 6))

    t_ext = np.zeros((np.shape(t)[0], 6))

    # Kinematic model initialization
    ur16e_Kinematic = Kinematic()
    q_actual = np.zeros(np.shape(qr))

    for i in range(np.shape(t)[0]):
        R, P, T = ur16e_Kinematic.TBF_Forward(qr[i, :])

        # R = np.transpose(cv2.Rodrigues(T[0:3, 0:3])[0])
        Q = mat2quat(R)
        xr[i, :] = np.hstack((P, Q))

        if (i > 0):
            # dt[i] = t[i] - t[i - 1]
            dt[i] = 0.01
            dqr[i - 1, :] = (qr[i, :] - qr[i - 1, :]) / dt[i]
            ddqr[i - 1, :] = (dqr[i, :] - dqr[i - 1, :]) / dt[i]
            J = jacobi.get_Jacobi(qr[i - 1, :])
            dxr[i - 1, :] = J.dot(dqr[i - 1, :])
            dxr[i - 1, 0:3] = (xr[i, 0:3] - xr[i - 1, 0:3]) / dt[i]
            J_dot = jacobi.get_Jacobi_dot(q=qr[i - 1, :], dq=dqr[i - 1, :])
            ddxr[i - 1, :] = J.dot(ddqr[i - 1, :]) + J_dot.dot(dqr[i - 1, :])
    for i in range(np.shape(t)[0] - 1):
        ddxr[i, 0:3] = (dxr[i + 1, 0:3] - dxr[i, 0:3]) / dt[i + 1]

        # print(i,ddxr[i-1,0:3],dxr[i,0:3],dxr[i-1,0:3])

    dt[0] = dt[1]
    x[0, :] = xr[0, :]
    dx[0, :] = dxr[0, :]
    ddx[0, :] = ddxr[0, :]

    z[0, :] = xr[0, 0:3]
    dz[0] = dxr[0, 0:3]
    ddz[0] = ddxr[0, 0:3]

    q[0, :] = qr[0, :]
    dq[0, :] = dqr[0, :]
    ddq[0, :] = ddqr[0, :]

    # mujoco controller

    ctrl = UR16e_Controller()  # 定义控制器

    ######################################'''joint space controller'''################################
    ####################################'''trajectory tracking controller'''#########################

    # ctrl.move_to_joint(qr[0,:],2,isRecord=False)
    #
    # IFC = ImpedanceForceController_Joint(Kd=ctrl.robot_controller.kd * np.eye(6),
    #                                      Ks=ctrl.robot_controller.kp * np.eye(6), Qd=0.003 * np.eye(6))
    # for i in range(np.shape(t)[0] - 1):
    #
    #     # Ks[i, :] = np.array([IFC.Ks[0, 0], IFC.Ks[1, 1], IFC.Ks[2, 2], IFC.Ks[3, 3], IFC.Ks[4, 4], IFC.Ks[5, 5]])
    #     Kd[i, :] = np.array([IFC.Kd[0, 0], IFC.Kd[1, 1], IFC.Kd[2, 2], IFC.Kd[3, 3], IFC.Kd[4, 4], IFC.Kd[5, 5]])
    #     tau[i, :] = ctrl.getForce()
    #
    #     # """ torque controller """
    #     ctrl.robot_controller.update()
    #
    #     ctrl.robot_controller.kd = IFC.Kd[0,0]
    #     ctrl.robot_controller.kp = IFC.Ks[0,0]
    #
    #     M_q = ctrl.robot_controller.mass_matrix
    #     # v[i, :] = M_q.dot(ddq[i,:])
    #     v[i, :] = M_q.dot(ddqr[i,:]+ ctrl.robot_controller.kd * (dqr[1,:]-ctrl.getQVel()) + ctrl.robot_controller.kp *
    #                       (qr[i, :] - ctrl.getQPos()))
    #
    #     w[i, :] = ctrl.robot_controller.torque_compensation
    #     t_ext[i, :] = v[i, :] + w[i, :]
    #     ctrl.move_to_torque(torque=t_ext[i, :],total_time=dt[i],isRecord=True)
    #
    #     """ position controller"""
    #     # ctrl.move_to_joint(qr[i, :], dt[i], isRecord=True)
    #
    #     q[i,:] = ctrl.getQPos()
    #     dq[i,:] = ctrl.getQVel()
    #     ddq[i,:]= ctrl.getQAcc()

    #     # IFC.Kd = ctrl.robot_controller.kd * np.eye(6)
    #     # IFC.Ks = ctrl.robot_controller.kp * np.eye(6)
    #
    #     # update controller parameters
    #     # IFC.update_Kd(q=q[i, :], qr=qr[i, :], dq=dq[i, :], dqr=dqr[i, :])

    ##########################'''joint space impedance controller'''##############################
    # ctrl.move_to_joint(qr[0, :], 5, isRecord=False)
    #
    # IFC = ImpedanceForceController_Joint(gamma= 0.1 *np.eye(6),Kd = np.diag([0.,0.,0.,0.,0.,0.]),
    #                                      Ks=0.* np.eye(6), Qs=60. * np.eye(6),Qd=0.03 * np.eye(6),
    #                                      tau=np.array([[0., 0., 0., 0., 0., 0.]]),Qf= 0.000036* np.eye(6),Qr = 0.0002 * np.eye(6),beta=0.)
    # for i in range(np.shape(t)[0]):
    #
    #     ks = np.array([IFC.Ks[0, 0], IFC.Ks[1, 1], IFC.Ks[2, 2], IFC.Ks[3, 3], IFC.Ks[4, 4], IFC.Ks[5, 5]])
    #     kd = np.array([IFC.Kd[0, 0], IFC.Kd[1, 1], IFC.Kd[2, 2], IFC.Kd[3, 3], IFC.Kd[4, 4], IFC.Kd[5, 5]])
    #
    #     ctrl.robot_controller.update()
    #     M_q = ctrl.robot_controller.mass_matrix
    #     CG_q = ctrl.robot_controller.torque_compensation
    #
    #     q[i, :] = ctrl.getQPos()
    #     dq[i, :] = ctrl.getQVel()
    #     ddq[i, :] = ctrl.getQAcc()
    #
    #     # J = jacobi.get_Jacobi(qr[i, :])
    #
    #     e, de, epsilon = IFC.compute_sliding_error(q=q[i, :], qr=qr[i, :], dq=dq[i, :], dqr=dqr[i, :])
    #
    #     # torque of robot
    #     # v[i, :] = M_q.dot(ddqr[i, :]- np.multiply(ctrl.robot_controller.kd, de))\
    #     #           + CG_q
    #     v[i, :] = M_q.dot(ddqr[i, :] + np.multiply(ctrl.robot_controller.kd, dqr[i - 1, :] - dq[i, :])+
    #                       np.multiply(ctrl.robot_controller.kp, qr[i - 1, :] - q[i, :])) + CG_q-IFC.gamma.dot(epsilon)
    #     # w[i, :] = -1. * (IFC.tau.reshape(6) + (IFC.Ks[0, 0] * (q[i, :]) + IFC.Kd[0, 0] * (dq[i, :])))
    #
    #     w[i, :] = -0.09 * (IFC.tau.reshape(6) + np.multiply(ks,e) + np.multiply(kd,de))
    #
    #     print('w:',w[i, :],'e:',e,'de:',de)
    #
    #     t_ext[i, :] = v[i, :] + w[i, :]
    #
    #     Ks[i, :] = np.array([IFC.Ks[0, 0], IFC.Ks[1, 1], IFC.Ks[2, 2], IFC.Ks[3, 3], IFC.Ks[4, 4], IFC.Ks[5, 5]])
    #     Kd[i, :] = np.array([IFC.Kd[0, 0], IFC.Kd[1, 1], IFC.Kd[2, 2], IFC.Kd[3, 3], IFC.Kd[4, 4], IFC.Kd[5, 5]])
    #     tau[i, :] = IFC.tau
    #
    #     ctrl.move_to_torque(t_ext[i, :], dt[i], isRecord=True)
    #
    #     # update controller parameters
    #     IFC.update_parameter(q=q[i, :], qr=qr[i, :], dq=dq[i, :], dqr=dqr[i, :], tau_d=-1. *tau_j[i, :])
    #
    #
    # plt.plot(tau, label='force')
    # plt.legend()
    # plt.show()
    # plt.plot(w, label='w')
    # plt.legend()
    # plt.show()
    #
    # # 开始画图
    # plt.figure(figsize=(9,6))
    # plt.title('Trajectory tracking based on PD controller')
    #
    # plt.plot(t, qr[:, 0],'--', label='reference q0')
    # plt.plot(t, qr[:, 1],'--', label='reference q1')
    # plt.plot(t, qr[:, 2],'--', label='reference q2')
    # plt.plot(t, qr[:, 3],'--', label='reference q3')
    # plt.plot(t, qr[:, 4],'--', label='reference q4')
    # plt.plot(t, qr[:, 5],'--', label='reference q5')
    #
    # plt.plot(t, q[:, 0],'-', label='set q0')
    # plt.plot(t, q[:, 1],'-', label='set q1')
    # plt.plot(t, q[:, 2],'-', label='set q2')
    # plt.plot(t, q[:, 3],'-', label='set q3')
    # plt.plot(t, q[:, 4],'-', label='set q4')
    # plt.plot(t, q[:, 5],'-', label='set q5')
    #
    # plt.legend(bbox_to_anchor=(0, 0), loc=3, borderaxespad=0) # 显示图例
    #
    # plt.xlabel('t(s)')
    # plt.ylabel('joint angle(rad)')
    # plt.show()

    ##########################'''Cartesian space impedance controller'''##############################
    """2018 RAS Adaptive variable imperance control for dynamic contact force tracking in uncertain environment"""
    ctrl.move_to_joint(qr[0, :], 5, isRecord=False)

    bb = np.zeros((len(t) - 1, 3))

    dz_E = np.zeros(len(t))
    ddz_E = np.zeros(len(t))

    dz_B = np.zeros(len(t))
    ddz_B = np.zeros(len(t))

    dz_E[0] = np.dot(ur16e_Kinematic.TBF_Forward(q=ctrl.getQPos())[0], dxr[0, 0:3])[2]
    dz_E[0] = np.dot(ur16e_Kinematic.TBF_Forward(q=ctrl.getQPos())[0], ddxr[0, 0:3])[2]

    m = 1.
    b = np.array([10., 10., 10.])
    phi = 0.
    sigma = 0.05

    for i in range(len(t) - 2):
        actual_q = ctrl.getQPos()
        actual_Fz = ctrl.getForce()[2]

        R, P, TBE = ur16e_Kinematic.TBF_Forward(q=actual_q)

        # dFz_E = actual_Fz + 25.
        dFz_E = actual_Fz - ft[i, 2]
        dFz = np.dot(R, [0, 0, dFz_E])

        # dxr_E = np.dot(np.linalg.pinv(R),dxr[i+1, 0:3])
        # ddxr_E = np.dot(np.linalg.pinv(R),ddxr[0, 0:3])
        #
        # ddz_E[i+1] = ddxr_E[2] + 1./m * (dFz_E - b[0]* (dz_E[i]- dxr_E[2]))
        # dz_E[i+1] = dz_E[i] + ddz_E[i+1] * dt[i+1]
        # z[i + 1, :] = z[i,0:3] + np.dot(R,np.array([0,0,dz_E[i+1]* dt[i+1]]))

        ddz[i + 1, :] = ddxr[i + 1, 0:3] + 1. / m * (dFz - np.multiply(b, (dz[i, :] - dxr[i + 1, 0:3])))
        dz[i + 1, 0:3] = dz[i, 0:3] + ddz[i + 1, 0:3] * dt[i + 1]
        z[i + 1, :] = z[i, 0:3] + dz[i + 1, 0:3] * dt[i + 1]

        desired_position = np.zeros(7)
        desired_position[0:3] = z[i + 1, :]
        desired_position[3:] = xr[i + 1, 3:]

        ctrl.move_to_point(desired_position, dt[i], isRecord=True)

        # phi = phi + sigma * (dFz / b)
        # b = b + b/(dz[i,:]- dxr[i+1, 0:3]) * phi

        print('phi:', phi, 'b:', b, 'Fz:', actual_Fz)

        bb[i, :] = b

    # plt.plot(bb)
    # plt.show()

    ##############################'''force-pose controller'''#################################

    # Pid1 = Pid(1, 0.002, 0.05, 0.001, 0.0001)
    # ctrl.move_to_joint(qr[0,:],2,isRecord=False)
    # for i in range(len(t)):
    #     actual_q = ctrl.getQPos()
    #     actual_Fz = ctrl.getForce()[2]///////////////////////////////
    #     actual_p = ctrl.getPose()
    #     R,P,TBE = ur16e_Kinematic.TBF_Forward(q=actual_q)
    #     # TEB = np.linalg.pinv(R)
    #     Xe = Pid1.pid_force_position(exp_fz= -1 * ft[i, 2], now_fz=actual_Fz)
    #     Xe = np.dot(R,[0, 0,-1 * Xe])
    #     # Xe = np.dot(R,[0, 0, 0.002])
    #     print('Fd:', -1 * ft[i, 2],'Fa:',actual_Fz,'Xe:',Xe)
    #     desired_position = np.zeros(7)
    #     # desired_position[0:3] = xr[i,0:3] + np.ndarray.flatten(np.array(Xe))
    #     desired_position[0:3] = xr[i, 0:3] + np.array(Xe)
    #     desired_position[3:] = xr[i,3:]
    #
    #     ctrl.move_to_point(desired_position,dt[i],isRecord=True)
    #     x[i,:] = ctrl.getPose()
    #     tau[i,:] = ctrl.getForce()

    # ######################### 测试关节控制  ############################
    # target = [1.57, -1.57, 1.57, -1.57, -1.57, -1.57]
    # ctrl.move_to_joint(target, total_time=10, isRecord=False)
    #
    # ctrl.stay(10)
    #
    # #######################  测试点控  ##########################
    # ctrl.move_to_point([-0.18056417, 0.47456404, 0.40209893, 3.141592653589793, 0.0, 0.0], 5, isRecord=False)
    # ctrl.stay(10)
    # pose = [-0.4 + 0.22, 0.6, -0.01, 3.141592653589793, 0.0, 0.0]
    # trajectory = []
    # for i in range(1000):
    #     pose[0] += 0.8 / 1000
    #     trajectory.append(deepcopy(pose))
    # pose[1] -= 0.2
    # trajectory.append(deepcopy(pose))
    # for i in range(1000):
    #     pose[0] -= 0.8 / 1000
    #     trajectory.append(deepcopy(pose))
    # pose[1] -= 0.2
    #
    # ##########################  测试轨迹控制 ###########################
    # ctrl.move_to_trajectory(trajectory, dt=0.02, isRecord=True)
    #
    # ########################### 轨迹信息保存 #####################
    # head = "px py pz qx qy qz qw " \
    #        "joint1 joint2 joint3 joint4 joint5 joint6 fx fy fz tx ty tz"
    # saveArray = np.hstack([ctrl.pose_list, ctrl.joint_position, ctrl.force_list])
    # ctrl.saveAsCSV(head, saveArray)
    #
    # ########################### 轨迹信息读取 #####################
    # array = ctrl.loadFromCSV("../data/20220411_144648.csv")
    # traject = array[:, 0:7]
    # ctrl.move_to_point(pos=traject[0], total_time=5)  # 先运动到第一个点
    # ctrl.move_to_trajectory(trajectory=traject)  # 复现轨迹

    l = ctrl.force_list
    l = np.array(l)
    length = [i for i in range(l.shape[0])]
    plt.plot(l[:, 2], 'r--', label='Fz')
    plt.plot(ft[:, 2], label='Fz_d')

    plt.xlabel("step")
    plt.ylabel("force_z(N)")
    plt.legend()
    plt.show()

    p = ctrl.pose_list
    p = np.array(p)
    length = [i for i in range(p.shape[0])]

    plt.figure(24, figsize=(9, 6))
    plt.subplot(2, 4, 1)
    plt.plot(length, p[:, 0])

    plt.xlabel("step")
    plt.ylabel("x(m)")
    plt.subplot(2, 4, 2)
    plt.plot(length, p[:, 1])

    plt.xlabel("step")
    plt.ylabel("y(m)")
    plt.subplot(2, 4, 3)
    plt.plot(length, p[:, 2])
    plt.xlabel("step")
    plt.ylabel("z         m")
    plt.subplot(2, 4, 4)
    plt.plot(length, p[:, 3])
    plt.xlabel("step")
    plt.ylabel("q1")
    plt.subplot(2, 4, 5)
    plt.plot(length, p[:, 4])
    plt.title("q2")
    plt.xlabel("step")
    plt.ylabel("q2")
    plt.subplot(2, 4, 6)
    plt.plot(length, p[:, 5])
    plt.title("q3")
    plt.xlabel("step")
    plt.ylabel("q3")
    plt.show()
    plt.subplot(2, 4, 7)
    plt.plot(length, p[:, 6])
    plt.title("q4")
    plt.xlabel("step")
    plt.ylabel("q4")
    plt.show()

    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 10,
    #          }
    #
    # l = ctrl.force_list
    # l = np.array(l)
    # length = [i for i in range(l.shape[0])]
    # plt.figure(23,figsize=(18,12))
    # plt.subplot(2, 3, 1)
    # plt.plot(length, l[:, 0])
    #
    # plt.xlabel("step",font2)
    # plt.ylabel("force_x(N)")
    # plt.subplot(2, 3, 2)
    # plt.plot(length, l[:, 1])
    #
    # plt.xlabel("step")
    # plt.ylabel("force_y(N)")
    # plt.subplot(2, 3, 3)
    # plt.plot(length, l[:, 2])
    # plt.plot(length, -1 * ft[:, 2], label='Fa')
    # plt.xlabel("step")
    # plt.ylabel("force_z(N)")
    # plt.subplot(2, 3, 4)
    # plt.plot(length, l[:, 3])
    # plt.xlabel("step")
    # plt.ylabel("torque_x(N*m)")
    # plt.subplot(2, 3, 5)
    # plt.plot(length, l[:, 4])
    # plt.title("ty")
    # plt.xlabel("step")
    # plt.ylabel("torque_y(N*m)")
    # plt.subplot(2, 3, 6)
    # plt.plot(length, l[:, 5])
    # plt.title("tz")
    # plt.xlabel("step")
    # plt.ylabel("torque_z(N*m)")
    # plt.show()


if __name__ == '__main__':
    main()
