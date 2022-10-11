#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/27
# @Author : Zzy
from matplotlib import pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
z = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体样式
plt.rcParams['font.size'] = 15  # 全局字体大小
plt.rcParams['axes.linewidth'] = 1


# 在坐标轴中画单个表
def data_plot(ax, x, y, xlabel, ylabel, title="", color='r', is_grid=False):
    ax.plot(x, y, color=color, linestyle='-.', linewidth=0.8)
    ax.set_title(title, fontsize=9, )  # 设置标题
    ax.spines['right'].set_visible(False)  # 设置右侧坐标轴不可见
    ax.spines['top'].set_visible(False)  # 设置上坐标轴不可见
    ax.spines['top'].set_linewidth(0.6)  # 设置坐标轴线宽
    ax.spines['right'].set_linewidth(0.6)  # 设置坐标轴线宽
    ax.set_xlabel(xlabel, fontsize=15, labelpad=5)  # 设置横轴字体和距离坐标轴的距离
    ax.set_ylabel(ylabel, fontsize=15, labelpad=5)  # 设置纵轴字体和距离坐标轴的距离
    # ax.set_ylim(0,10000)  #设置y轴范围
    # ax.set_xlim(0, 10000)  # 设置x轴范围
    # 添加网格
    if is_grid:
        ax.grid(which='major', ls='--', alpha=.8, lw=.5)  # 是否设置网格，设置网格宽度和形状
    # 设置刻度坐标的朝向
    # ax.tick_params(which='major', x=5, width=1.5, direction='in', top='on', right="on")
    # ax.tick_params(which='minor', x=3, width=1, direction='in', top='on', right="on")


# 新建图表
fig = plt.figure(figsize=(16, 10))

# 设置整个图表之间的距离
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                    wspace=0.4, hspace=0.4)

fig.suptitle('x - y', fontsize=20)
ax1 = fig.add_subplot(231)
data_plot(ax1, x=x, y=y, xlabel="step", ylabel="force_x  N", title="X", is_grid=True)

ax2 = fig.add_subplot(232)
data_plot(ax2, x, y, "step", "force_x  N")

ax3 = fig.add_subplot(233)
data_plot(ax3, x, y, "step", "force_x  N")

ax4 = fig.add_subplot(234)
data_plot(ax4, x, y, "step", "force_x  N")

ax5 = fig.add_subplot(235)
data_plot(ax5, x, y, "step", "force_x  N")

ax6 = fig.add_subplot(236)
data_plot(ax6, x, y, "step", "force_x  N")
plt.savefig(r'./test.jpg', dpi=300, bbox_inches='tight')

# 新建另一个图表
fig1 = plt.figure(figsize=(16, 10))
ax7 = fig1.add_subplot(111)  # 只有一个表
ax7.plot(x, y, color='b', linestyle='-.', linewidth=0.8, label="x-y")
ax7.plot(x, z, color='r', linestyle='-.', linewidth=0.8, label="x-z")
# 添加图例

# 设置图例的位置，loc 后的数字一般取 1、2、3、4，依次表示右上、左上、左下和右下
# prop 表示字体属性，我们选择预设的 edgecolor 表示图例边框颜色，w 为白色；framealpha 后的数字表示透明度
# ncol=2 控制图例展示的列数；bbox_to_anchor=(x,y) 控制图例的位置。x 和 y 值为相对于坐标区的比例，依次为左右和上下
plt.legend(loc=4, fontsize=15, edgecolor='b', framealpha=0, ncol=1, bbox_to_anchor=(0.85, 0.9), title="x-")
ax7.set_title("x-y z", fontsize=15, pad=15)  # 设置标题
ax7.spines['right'].set_visible(False)  # 设置右侧坐标轴不可见
ax7.spines['top'].set_visible(False)  # 设置上坐标轴不可见
ax7.spines['top'].set_linewidth(0.6)  # 设置坐标轴线宽
ax7.spines['right'].set_linewidth(0.6)  # 设置坐标轴线宽
ax7.set_xlabel("x", fontsize=15, labelpad=5)  # 设置横轴字体和距离坐标轴的距离
ax7.set_ylabel("y - z", fontsize=15, labelpad=5)  # 设置纵轴字体和距离坐标轴的距离
# ax.set_ylim(0,10000)  #设置y轴范围
# ax.set_xlim(0, 10000)  # 设置x轴范围
# 添加网格
# ax7.grid(which='major', ls='--', alpha=.8, lw=.5)  # 是否设置网格，设置网格宽度和形状
# 设置刻度坐标的朝向
# ax.tick_params(which='major', x=5, width=1.5, direction='in', top='on', right="on")
# ax.tick_params(which='minor', x=3, width=1, direction='in', top='on', right="on")


plt.show()
