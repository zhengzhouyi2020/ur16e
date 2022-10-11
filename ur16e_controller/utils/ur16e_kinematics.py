# -*- coding: utf-8 -*-
# @Author  : zzy
# @FileName: ur16e_kinematics.py
# @Time    : 2021/3/9
# @Description:  正逆运动学求解

import math
import numpy as np

#### 这里不考虑底板1.5mm的高度
# d (unit: m)
d1 = 0.1807
d2 = d3 = 0
d4 = 0.17415
d5 = 0.11985
d6 = 0.11655
d_effort = 0.149
# d_effort = 0.34163
# d_effort = 0.32663
# a (unit: m)
a1 = a4 = a5 = a6 = 0
a2 = -0.4784
a3 = -0.36
PI = math.pi
ZERO_THRESH = 0.00000001


def Forward(q):
    s1 = math.sin(q[0])
    c1 = math.cos(q[0])
    # q02
    q23 = q[1]
    q234 = q[1]
    s2 = math.sin(q[1])
    c2 = math.cos(q[1])
    # q03
    s3 = math.sin(q[2])
    c3 = math.cos(q[2])
    q23 += q[2]
    q234 += q[2]
    # q04
    q234 += q[3]
    # q05
    s5 = math.sin(q[4])
    c5 = math.cos(q[4])
    # q06
    s6 = math.sin(q[5])
    c6 = math.cos(q[5])
    s23 = math.sin(q23)
    c23 = math.cos(q23)
    s234 = math.sin(q234)
    c234 = math.cos(q234)
    T = [0] * 16
    T[0] = c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6
    T[1] = - s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6
    T[2] = c5 * s1 - c234 * c1 * s5
    T[3] = d6 * (c5 * s1 - c234 * c1 * s5) + c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1
    T[4] = - c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6
    T[5] = s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1
    T[6] = - c1 * c5 - c234 * s1 * s5
    T[7] = s1 * (a3 * c23 + a2 * c2) - d4 * c1 - d6 * (c1 * c5 + c234 * s1 * s5) + d5 * s234 * s1
    T[8] = c234 * s6 + s234 * c5 * c6
    T[9] = c234 * c6 - s234 * c5 * s6
    T[10] = -s234 * s5
    T[11] = d1 + a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5
    T[12] = 0
    T[13] = 0
    T[14] = 0
    T[15] = 1
    return T


def SIGN(x):
    if x < 0:
        return - 1
    return 1


def inverse(T, q6_des=0.0):
    """
    运动学反解,
    :param T: 当前位置矩阵
    :param q6_des: 目标关节角度值
    :return:
    num_sols:解的个数
    q_sols: 解集  6*8=48一阶矩阵
    """
    q_sols = [0] * 48
    num_sols = 0
    T00 = T[0][0]
    T01 = T[0][1]
    T02 = T[0][2]
    T03 = T[0][3]
    T10 = T[1][0]
    T11 = T[1][1]
    T12 = T[1][2]
    T13 = T[1][3]
    T20 = T[2][0]
    T21 = T[2][1]
    T22 = T[2][2]
    T23 = T[2][3]

    #####shoulder rotate  joint(q1) ###########################
    q1 = [0, 0]
    A = d6 * T12 - T13
    B = d6 * T02 - T03
    R = A * A + B * B
    if math.fabs(A) < ZERO_THRESH:
        if math.fabs(math.fabs(d4) - math.fabs(B)) < ZERO_THRESH:
            div = - SIGN(d4) * SIGN(B)
        else:
            div = -d4 / B
        arcsin = math.asin(div)
        if math.fabs(arcsin) < ZERO_THRESH:
            arcsin = 0.0
        if arcsin < 0.0:
            q1[0] = arcsin + 2.0 * PI
        else:
            q1[0] = arcsin
            q1[1] = PI - arcsin
    elif math.fabs(B) < ZERO_THRESH:
        if math.fabs(math.fabs(d4) - math.fabs(A)) < ZERO_THRESH:
            div = SIGN(d4) * SIGN(A)
        else:
            div = d4 / A
        arccos = math.acos(div)
        q1[0] = arccos
        q1[1] = 2.0 * PI - arccos
    elif d4 * d4 > R:
        return num_sols
    else:
        arccos = math.acos(d4 / math.sqrt(R))
        arctan = math.atan2(-B, A)
        pos = arccos + arctan
        neg = -arccos + arctan
        if math.fabs(pos) < ZERO_THRESH:
            pos = 0.0
        if math.fabs(neg) < ZERO_THRESH:
            neg = 0.0
        if pos >= 0.0:
            q1[0] = pos
        else:
            q1[0] = 2.0 * PI + pos
        if neg >= 0.0:
            q1[1] = neg
        else:
            q1[1] = 2.0 * PI + neg

    ###### wrist2  joint(q5) ##########################
    q5 = np.zeros((2, 2))  # define 2*2 q5 array

    for i in range(2):
        numer = (T03 * math.sin(q1[i]) - T13 * math.cos(q1[i]) - d4)
        if math.fabs(math.fabs(numer) - math.fabs(d6)) < ZERO_THRESH:
            div = SIGN(numer) * SIGN(d6)
        else:
            div = numer / d6
        arccos = math.acos(div)
        q5[i][0] = arccos
        q5[i][1] = 2.0 * PI - arccos
    #############################################################
    for i in range(2):
        for j in range(2):
            c1 = math.cos(q1[i])
            s1 = math.sin(q1[i])
            c5 = math.cos(q5[i][j])
            s5 = math.sin(q5[i][j])
            ######################## wrist 3 joint (q6) ################################
            if math.fabs(s5) < ZERO_THRESH:
                q6 = q6_des
            else:
                q6 = math.atan2(SIGN(s5) * -(T01 * s1 - T11 * c1), SIGN(s5) * (T00 * s1 - T10 * c1))
                if math.fabs(q6) < ZERO_THRESH:
                    q6 = 0.0
                if (q6 < 0.0):
                    q6 += 2.0 * PI
            q2 = [0, 0]
            q3 = [0, 0]
            q4 = [0, 0]
            #####################RRR joints (q2, q3, q4) ################################
            c6 = math.cos(q6)
            s6 = math.sin(q6)
            x04x = -s5 * (T02 * c1 + T12 * s1) - c5 * (s6 * (T01 * c1 + T11 * s1) - c6 * (T00 * c1 + T10 * s1))
            x04y = c5 * (T20 * c6 - T21 * s6) - T22 * s5
            p13x = d5 * (s6 * (T00 * c1 + T10 * s1) + c6 * (T01 * c1 + T11 * s1)) - d6 * (
                    T02 * c1 + T12 * s1) + T03 * c1 + T13 * s1
            p13y = T23 - d1 - d6 * T22 + d5 * (T21 * c6 + T20 * s6)
            c3 = (p13x * p13x + p13y * p13y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            if math.fabs(math.fabs(c3) - 1.0) < ZERO_THRESH:
                c3 = SIGN(c3)
            elif math.fabs(c3) > 1.0:
                # TODO NO SOLUTION
                continue
            arccos = math.acos(c3)
            q3[0] = arccos
            q3[1] = 2.0 * PI - arccos
            denom = a2 * a2 + a3 * a3 + 2 * a2 * a3 * c3
            s3 = math.sin(arccos)
            A = (a2 + a3 * c3)
            B = a3 * s3
            q2[0] = math.atan2((A * p13y - B * p13x) / denom, (A * p13x + B * p13y) / denom)
            q2[1] = math.atan2((A * p13y + B * p13x) / denom, (A * p13x - B * p13y) / denom)
            c23_0 = math.cos(q2[0] + q3[0])
            s23_0 = math.sin(q2[0] + q3[0])
            c23_1 = math.cos(q2[1] + q3[1])
            s23_1 = math.sin(q2[1] + q3[1])
            q4[0] = math.atan2(c23_0 * x04y - s23_0 * x04x, x04x * c23_0 + x04y * s23_0)
            q4[1] = math.atan2(c23_1 * x04y - s23_1 * x04x, x04x * c23_1 + x04y * s23_1)
            ###########################################
            for k in range(2):
                if math.fabs(q2[k]) < ZERO_THRESH:
                    q2[k] = 0.0
                elif q2[k] < 0.0:
                    q2[k] += 2.0 * PI
                if math.fabs(q4[k]) < ZERO_THRESH:
                    q4[k] = 0.0
                elif q4[k] < 0.0:
                    q4[k] += 2.0 * PI
                q_sols[num_sols * 6 + 0] = q1[i]
                q_sols[num_sols * 6 + 1] = q2[k]
                q_sols[num_sols * 6 + 2] = q3[k]
                q_sols[num_sols * 6 + 3] = q4[k]
                q_sols[num_sols * 6 + 4] = q5[i][j]
                q_sols[num_sols * 6 + 5] = q6
                num_sols += 1
    if num_sols > 0:
        for i in range(num_sols):
            for j in range(6):
                if q_sols[i * 6 + j] > PI:
                    q_sols[i * 6 + j] -= 2 * PI
                elif q_sols[i * 6 + j] < -PI:
                    q_sols[i * 6 + j] += 2 * PI
                else:
                    continue
    return num_sols, q_sols


def IKSelection(num_sols, q_sol, jnt_init):
    jnt_out = [0] * 6
    q_diff = [0] * 8
    min_dis = 100
    min_index = 0
    if num_sols > 0:
        for i in range(num_sols):
            for j in range(6):
                diff_temp = q_sol[i * 6 + j] - jnt_init[j]
                if diff_temp > PI and q_sol[i * 6 + j] > 0:  ## 是否满足
                    q_sol[i * 6 + j] -= 2 * PI
                    diff_temp -= 2.0 * PI
                elif diff_temp < -PI and q_sol[i * 6 + j] < 0:
                    q_sol[i * 6 + j] += 2 * PI
                    diff_temp += 2.0 * PI
                q_diff[i] += abs(diff_temp)
            if q_diff[i] < min_dis:
                min_dis = q_diff[i]
                min_index = i
    for i in range(6):
        jnt_out[i] = q_sol[min_index * 6 + i]
    return jnt_out


def get_Jacobi(q):
    d6_effort = d6 + d_effort  # 需要加上末端的长度
    s1 = math.sin(q[0])
    c1 = math.cos(q[0])
    # q02
    q23 = q[1]
    q234 = q[1]
    s2 = math.sin(q[1])
    c2 = math.cos(q[1])
    # q03
    s3 = math.sin(q[2])
    c3 = math.cos(q[2])
    q23 += q[2]
    q234 += q[2]
    # q04
    q234 += q[3]
    # q05
    s5 = math.sin(q[4])
    c5 = math.cos(q[4])
    # q06
    s6 = math.sin(q[5])
    c6 = math.cos(q[5])
    s23 = math.sin(q23)
    c23 = math.cos(q23)
    s234 = math.sin(q234)
    c234 = math.cos(q234)
    # 数组都是行储存的
    jacp = np.zeros([3, 6])
    jacr = np.zeros([3, 6])

    jacp[0, 0] = d6_effort * (c1 * c5 + s1 * c234 * s5) - d5 * s1 * s234 + d4 * c1 - a3 * s1 * c23 - a2 * s1 * c2
    jacp[0, 1] = d6_effort * s1 * c234 * s5 + d5 * c1 * c234 - a3 * c1 * s23 - a2 * c1 * s2
    jacp[0, 2] = d6_effort * c1 * s234 * s5 + d5 * c1 * c234 - a3 * c1 * s23
    jacp[0, 3] = d6_effort * c1 * s234 * s5 + d5 * c1 * c234
    jacp[0, 4] = -d6_effort * (s1 * s5 + c1 * c234 * c5)
    jacp[0, 5] = 0

    jacp[1, 0] = d6_effort * (s1 * c5 - c1 * c234 * s5) + d5 * c1 * s234 + d4 * s1 + a3 * c1 * c23 + a2 * c1 * c2
    jacp[1, 1] = d6_effort * s1 * s234 * s5 + d5 * s1 * c234 - a3 * s1 * s23 - a2 * s1 * s2
    jacp[1, 2] = d6_effort * s1 * s234 * s5 + d5 * s1 * c234 - a3 * s1 * s23
    jacp[1, 3] = d6_effort * s1 * s234 * s5 + d5 * s1 * c234
    jacp[1, 4] = d6_effort * (c1 * s5 - s1 * c234 * s5)
    jacp[1, 5] = 0

    jacp[2, 0] = 0
    jacp[2, 1] = a2 * c2 + a3 * c23 + d5 * s234 - d6_effort * c234 * s5
    jacp[2, 2] = a3 * c23 + d5 * s234 - d6_effort * c234 * s5
    jacp[2, 3] = d5 * s234 - d6_effort * c234 * s5
    jacp[2, 4] = -d6_effort * c234 * c5
    jacp[2, 5] = 0

    jacr[0, 0] = 0
    jacr[0, 1] = s1
    jacr[0, 2] = s1
    jacr[0, 3] = s1
    jacr[0, 4] = c1 * s234
    jacr[0, 5] = s1 * c5 - c1 * s5 * c234

    jacr[1, 0] = 0
    jacr[1, 1] = -c1
    jacr[1, 2] = -c1
    jacr[1, 3] = -c1
    jacr[1, 4] = s1 * s234
    jacr[1, 5] = -c1 * c5 - s1 * s5 * c234

    jacr[2, 0] = 1
    jacr[2, 1] = 0
    jacr[2, 2] = 0
    jacr[2, 3] = 0
    jacr[2, 4] = -c234
    jacr[2, 5] = -s5 * s234

    return np.vstack((jacp, jacr))


class Kinematic:
    """
    正逆运动学考虑了基座和末端执行器的长度
    """

    def __init__(self):
        global d1
        global a2
        global a3
        global d4
        global d5
        global d6
        global PI
        global ZERO_THRESH

        self.baseFrame = np.array([[1.0, 0, 0, 0],
                                   [0, 1.0, 0, 0],
                                   [0, 0, 1.0, 0],  ## 这里忽略底板1.5mm的高度
                                   [0, 0, 0, 1.0]])
        self.toolFrame = np.array([[1.0, 0, 0, 0],
                                   [0, 1.0, 0, 0],
                                   [0, 0, 1.0, d_effort],
                                   [0, 0, 0, 1.0]])

    def FKine(self, jnt_in):
        T = Forward(jnt_in)
        T = np.array(T).reshape(4, 4)
        frame_out = np.matmul((np.matmul(self.baseFrame, T)), self.toolFrame)
        return frame_out

    def IKine(self, T, jnt_init):
        frame_temp = np.matmul(np.matmul(np.linalg.inv(self.baseFrame), T), np.linalg.inv(self.toolFrame))
        num_sols, q_sols = inverse(frame_temp)
        jnt_out = IKSelection(num_sols, q_sols, jnt_init)
        return jnt_out


"""
将位姿四元数装化为齐次变换矩阵
"""


def rotateVector2quaternion(vector):
    theta = np.linalg.norm(vector[3:])
    kx = vector[3] / theta
    ky = vector[4] / theta
    kz = vector[5] / theta

    x = kx * np.sin(theta / 2)
    y = ky * np.sin(theta / 2)
    z = kz * np.sin(theta / 2)
    w = np.cos(theta / 2)
    return [vector[0], vector[1], vector[2], x, y, z, w]


def quaternion2rotateVector(orientation):
    k = np.linalg.norm(orientation[3:])  # 需要将四元数正则化
    x = orientation[3] / k
    y = orientation[4] / k
    z = orientation[5] / k
    w = orientation[6] / k

    theta = 2 * math.acos(w)
    kx = x / np.sqrt(1 - w * w)
    ky = y / np.sqrt(1 - w * w)
    kz = z / np.sqrt(1 - w * w)
    rx = theta * kx
    ry = theta * ky
    rz = theta * kz
    return [orientation[0], orientation[1], orientation[2], rx, ry, rz]


def pose2Transform(pose):
    """
    pose 包含[position,orientation]位姿四元数
    :param pose:当前位姿
    :return:齐次变换矩阵 (4,4)
    """
    k = np.linalg.norm(pose[3:])  # 需要将四元数正则化
    x = pose[3] / k
    y = pose[4] / k
    z = pose[5] / k
    w = pose[6] / k
    R = np.zeros((4, 4))
    R[0][0] = w * w + x * x - y * y - z * z
    R[0][1] = 2 * x * y - 2 * w * z
    R[0][2] = 2 * x * z + 2 * w * y
    R[1][0] = 2 * x * y + 2 * w * z
    R[1][1] = w * w - x * x + y * y - z * z
    R[1][2] = 2 * y * z - 2 * w * x
    R[2][0] = 2 * x * z - 2 * w * y
    R[2][1] = 2 * y * z + 2 * w * x
    R[2][2] = w * w - x * x - y * y + z * z
    R[0][3] = pose[0]
    R[1][3] = pose[1]
    R[2][3] = pose[2]
    R[3][3] = 1
    return R


def transform2Pose(T):
    """
    将齐次变换矩阵转化为位姿四元数
    :param T:
    :return: [1,7]的向量,包含[position,orientation]
    position:[x,y,z]
    orientation:[x,y,z,w]
    """

    w = np.sqrt(T[0][0] + T[1][1] + T[2][2] + 1) / 2
    x = np.sqrt(T[0][0] - T[1][1] - T[2][2] + 1) / 2
    y = np.sqrt(-T[0][0] + T[1][1] - T[2][2] + 1) / 2
    z = np.sqrt(-T[0][0] - T[1][1] + T[2][2] + 1) / 2
    a = [w, x, y, z]
    m = a.index(max(a))
    if m == 0:
        x = (T[2][1] - T[1][2]) / (4 * w)
        y = (T[0][2] - T[2][0]) / (4 * w)
        z = (T[1][0] - T[0][1]) / (4 * w)
    if m == 1:
        w = (T[2][1] - T[1][2]) / (4 * x)
        y = (T[0][1] + T[1][0]) / (4 * x)
        z = (T[2][0] + T[0][2]) / (4 * x)
    if m == 2:
        w = (T[0][2] - T[2][0]) / (4 * y)
        x = (T[0][1] + T[1][0]) / (4 * y)
        z = (T[1][2] + T[2][1]) / (4 * y)
    if m == 3:
        w = (T[1][0] - T[0][1]) / (4 * z)
        x = (T[2][0] + T[0][2]) / (4 * z)
        y = (T[1][2] + T[2][1]) / (4 * z)
    b = [T[0][3], T[1][3], T[2][3], x, y, z, w]
    return b


if __name__ == '__main__':
    # quaternion = [0.174743, 0.479965, 0.409409, 1,0,0,0]
    # print((pose_transform_T(quaternion)))
    # joint=[1.57,-1.57,-1.57,-1.57,1.57,0]
    # T=pose_transform_T(quaternion)
    # # joint = [0,0,0,0,0,0]
    # ur16e = Kinematic()
    # # T=ur16e.FKine(joint)
    # # print(T)
    # print(T_transform_pose(T))
    # q_out=ur16e.IKine(T,joint)
    # vector =quaternion2rotateVector(quaternion)
    # print(vector)
    # print(rotateVector2quaternion(vector))
    # orientation = rotateVector2quaternion(vector)
    # T = pose_transform_T(orientation)
    # jnt_out = ur16e.IKine(T, ur16e.IKine(T, [1.57,-1.57,-1.57,-1.57,1.57,0]))
    # print(jnt_out)

    ur16e = Kinematic()
    jnt1 = [-2.267876866030676, -0.2821222200643234, -1.5677169457978568, 0.27904283906728394, -1.5707963267948966,
            -0.6970805392357793]
    T1 = ur16e.FKine(jnt1)
    jnt2 = [1.57, -1.57, -1.57, -1.57, 1.57, 0]
    T2 = ur16e.FKine(jnt2)
    print(T1)
    print(T2)
