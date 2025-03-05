import numpy as np
from matplotlib import pyplot as plt
# path = np.loadtxt("path_7000.txt")
# start = np.loadtxt("x_init.txt")
# goal = np.loadtxt("x_goal.txt")
# plt.figure()
# plt.scatter(path[:,0],path[:,1])

# # highlight
# plt.scatter(start[:, 0],start[:, 1], c='r', linewidths=5)
# plt.scatter(goal[:, 0],goal[:, 1], c='b', linewidths=5)

# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 使用 np.loadtxt 读取二维数组
array = np.loadtxt('feasibility_2d.txt', dtype=int)
# start = np.loadtxt("start0.txt")
# goal = np.loadtxt("goal.t0xt")
# 收集红色点和蓝色点的坐标
red_points = [(j, i) for i in range(array.shape[0]) for j in range(array.shape[1]) if array[i, j] == 1]
blue_points = [(j, i) for i in range(array.shape[0]) for j in range(array.shape[1]) if array[i, j] == 0]

# 将坐标拆分为 x 和 y 列表
red_x, red_y = zip(*red_points) if red_points else ([], [])
blue_x, blue_y = zip(*blue_points) if blue_points else ([], [])

# 创建一个画布
plt.figure()

# 一次性绘制红色点和蓝色点
plt.scatter(red_x, red_y, color='red', label='1')
plt.scatter(blue_x, blue_y, color='blue', label='0')
# plt.scatter(start[0], start[1], c='green', linewidths=5)
# plt.scatter(goal[0], goal[1], c='green', linewidths=5)


# 反转y轴方向，使得数组的第0行在顶部
plt.gca().invert_yaxis()

# 添加图例
plt.legend()

# 显示图像
plt.title("2D Array Visualization")
plt.show()