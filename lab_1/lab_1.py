import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Дані (n = 20)
# Вхідні параметри
x1 = np.array([0, 0, 0, 1, 1, 2, 2, 2])
x2 = np.array([1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5])
# Обчислення вихідних значень згідно 20 варіанту:
# спостереження:
# 1: y = 2.3
# 2: y = 4 + 0.3*n = 4 + 0.3*20 = 10
# 3: y = 2 - 0.1*n = 2 - 0.1*20 = 0
# 4: y = 5 - 0.2*n = 5 - 0.2*20 = 1
# 5: y = 4 - 0.2*n = 4 - 0.2*20 = 0
# 6: y = 6.1 + 0.2*n = 6.1 + 0.2*20 = 10.1
# 7: y = 6.5 - 0.1*n = 6.5 - 0.1*20 = 4.5
# 8: y = 7.2
y = np.array([2.3, 10.0, 0.0, 1.0, 0.0, 10.1, 4.5, 7.2])

# Формуємо матрицю дизайн-матрицю X: стовпець з 1, x1 та x2
X = np.column_stack((np.ones(x1.shape[0]), x1, x2))

# Обчислюємо коефіцієнти за нормальними рівняннями: a = (X^T X)^(-1) X^T y
coeff = np.linalg.inv(X.T @ X) @ (X.T @ y)
a0, a1, a2 = coeff

print("Обчислені коефіцієнти:")
print(f"a0 = {a0:.4f}, a1 = {a1:.4f}, a2 = {a2:.4f}")

# Обчислення передбачених значень
y_pred = X @ coeff

# Обчислення коефіцієнта детермінації R^2
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
R2 = 1 - ss_res/ss_tot

print(f"Коефіцієнт детермінації R^2 = {R2:.4f}")

# Обчислення значення функції у точці x1 = 1.5, x2 = 3
x1_test, x2_test = 1.5, 3.0
y_test = a0 + a1 * x1_test + a2 * x2_test
print(f"Значення функції у точці (x1, x2) = ({x1_test}, {x2_test}) -> y = {y_test:.4f}")

# Побудова 3D графіку
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Точки спостереження
ax.scatter(x1, x2, y, color='red', label='Дані')

# Побудова сітки для площини апроксимації
x1_grid, x2_grid = np.meshgrid(np.linspace(min(x1)-0.5, max(x1)+0.5, 20),
                               np.linspace(min(x2)-0.5, max(x2)+0.5, 20))
y_grid = a0 + a1 * x1_grid + a2 * x2_grid

ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, label='Апроксимаційна площина')
# Для 3D поверхні plot_surface не підтримує параметр label, тому додамо легенду вручну
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Залежність y = a0 + a1*x1 + a2*x2')

# Створення легенди
red_proxy = plt.Line2D([0],[0], linestyle="none", marker='o', color='red')
blue_proxy = plt.Rectangle((0,0),1,1,fc="blue", alpha=0.5)
ax.legend([red_proxy, blue_proxy], ['Дані', 'Апроксимаційна площина'])

plt.show()