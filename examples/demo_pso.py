def demo_func(x):
    x1, x2 = x
    return x1 ** 2 + x2 ** 2 + abs(x1 * x2)


# %% Do PSO
from sko.PSO import PSO

pso = PSO(func=demo_func, n_dim=2, pop=40, max_iter=150, lb=[-10, -10], ub=[10, 10], w=0.8, c1=0.5, c2=0.5,
          verbose=True)
pso.run(precision=0.0005)
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

# %% Plot the result
import matplotlib.pyplot as plt

plt.plot(pso.gbest_y_hist)
plt.show()
