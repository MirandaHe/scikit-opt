import numpy as np
from sko.PSO import PSO


def demo_func(x):
    x1, x2 = x
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.e


# constraint_ueq = (
#     lambda x: (x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2
#     ,
# )

max_iter = 50
# pso = PSO(func=demo_func, n_dim=2, pop=40, max_iter=max_iter, lb=[-2, -2], ub=[2, 2]
#           , constraint_ueq=constraint_ueq)
pso = PSO(func=demo_func, n_dim=2, pop=40, max_iter=max_iter, lb=[-2, -2], ub=[2, 2])
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

# %% Now Plot the animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

record_value = pso.record_value
X_list, V_list = record_value['X'], record_value['V']

fig, ax = plt.subplots(1, 1)
ax.set_title('title', loc='center')
# 使用蓝色点状绘制粒子
line = ax.plot([], [], 'b.')

X_grid, Y_grid = np.meshgrid(np.linspace(-2.0, 2.0, 40), np.linspace(-2.0, 2.0, 40))
Z_grid = demo_func((X_grid, Y_grid))
# ax.contour用于绘制轮廓
ax.contour(X_grid, Y_grid, Z_grid, 30)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# 绘制约束区域，从0 到 2*pi 之间绘制40个序列点
# t = np.linspace(0, 2 * np.pi, 40)
# ax.plot(0.5 * np.cos(t) + 1, 0.5 * np.sin(t), color='r')

# plt.ion()
# plt.show()


def update_scatter(frame):
    print('frame', frame)
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    # 更新位置
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    # 给粒子设置坐标x,y的值
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line


# interval:更新频率, blit表示更新所有点
ani = FuncAnimation(fig, update_scatter, blit=True, interval=15, frames=max_iter * 10)
plt.show()
# ani.save('pso.gif', writer='pillow')
