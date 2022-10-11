#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/8
# @Author : Zzy
import numpy as np
import matplotlib.pyplot as plt

m20b100k10 = np.loadtxt("m20b100k10.csv")
m50b100k10 = np.loadtxt("m50b100k10.csv")
m100b100k10 = np.loadtxt("m100b100k10.csv")
m200b100k10 = np.loadtxt("m200b100k10.csv")
m500b100k10 = np.loadtxt("m500b100k10.csv")

m200b50k10 = np.loadtxt("m200b50k10.csv")
m200b100k10 = np.loadtxt("m200b100k10.csv")
m200b200k10 = np.loadtxt("m200b200k10.csv")
m200b500k10 = np.loadtxt("m200b500k10.csv")

m200b500k10 = np.loadtxt("m200b500k10.csv")
m200b500k50 = np.loadtxt("m200b500k50.csv")
m200b500k100 = np.loadtxt("m200b500k100.csv")
m200b500k200 = np.loadtxt("m200b500k200.csv")
m200b500k500 = np.loadtxt("m200b500k500.csv")

length = len(m20b100k10)
x = [i for i in range(length)]
plt.rcParams.update({'font.size': 15})
plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以plt绘图过程中中文无法显示的问题
plt.xlabel("step", fontsize=12, labelpad=5)
plt.ylabel('Force /N', fontsize=15, labelpad=5)
# plt.plot(x,m20b100k10,lw = 0.8,label = 'm20')
# plt.plot(x,m50b100k10,lw = 0.8,label = 'm50')
# plt.plot(x,m100b100k10,lw = 0.8,label = 'm100')
# plt.plot(x,m200b100k10,lw = 0.8,label = 'm200')
# plt.plot(x,m500b100k10,lw = 0.8,label = 'm500')

# plt.plot(x,m200b50k10,lw = 0.8,label = 'b50')
# plt.plot(x,m200b100k10,lw = 0.8,label = 'b100')
# plt.plot(x,m200b200k10,lw = 0.8,label = 'b200')
# plt.plot(x,m200b500k10,lw = 0.8,label = 'b500')

plt.plot(x, m200b500k10, lw=0.8, label='k20')
plt.plot(x, m200b500k50, lw=0.8, label='k50')
plt.plot(x, m200b500k100, lw=0.8, label='k100')
plt.plot(x, m200b500k200, lw=0.8, label='k200')
plt.plot(x, m200b500k500, lw=0.8, label='k500')
plt.legend()  # 显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()  # 显示图片，如果注释改行，即使设置了图片仍然不显示
