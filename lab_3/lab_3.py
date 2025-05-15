import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Завдання 1: Чисельне розв'язання (Runge-Kutta 4-го порядку)
# Вхідні дані
a = 23.6e-6        # м^2/с
L = 1.2            # м
T = 2 * 3600       # 2 години в секундах
N = 100            # кількість шарів
h = 0.3            # крок по часу (с)
alpha = 20         # ліва температура (°C)
beta = 39          # права температура (°C)
phi0 = 0           # початкова температура

delta = L / N
mu = a / delta**2
M = int(T / h)
# координати і час
y_values = np.linspace(0, L, N+1)
t_values = np.arange(0, T+h, h)

# Загальна матриця температури
u = np.zeros((M+1, N+1))
# Граничні умови
u[:, 0] = alpha
u[:, -1] = beta

# Функція похідних для внутрішніх вузлів
def du_dt(u_prev):
    du = np.zeros_like(u_prev)
    for i in range(1, N):
        du[i] = mu * (u_prev[i-1] - 2*u_prev[i] + u_prev[i+1])
    return du

def runge_kutta_step(u_prev):
    k1 = du_dt(u_prev)
    k2 = du_dt(u_prev + 0.5*h*k1)
    k3 = du_dt(u_prev + 0.5*h*k2)
    k4 = du_dt(u_prev + h*k3)
    return u_prev + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

for n in range(M):
    u[n+1] = runge_kutta_step(u[n])

# Завдання 2: Аналітичне рішення та порівняння
# Аналітичний розв'язок (30 доданків)

def analytical_solution(t, y, terms=30):
    result = ((beta - alpha) / L) * y + alpha
    for n in range(1, terms + 1):
        lam = np.pi * n / L
        coef = (2 / np.pi) * (1 / n) * (beta * ((-1)**n) - alpha)
        result += coef * np.sin(lam * y) * np.exp(-a * t * lam**2)
    return result

# Обчислення аналітичного рішення
u_analytical = np.zeros_like(u)
for idx, t in enumerate(t_values):
    u_analytical[idx] = analytical_solution(t, y_values)

# Обчислення похибок
abs_error = np.abs(u - u_analytical)
mae = np.max(abs_error)
mse = np.mean(abs_error**2)
print(f"MAE = {mae:.5f} °C")
print(f"MSE = {mse:.5f} °C^2")

# Візуалізація результатів у 3D
X, Y = np.meshgrid(y_values, t_values)
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u, cmap='viridis')
ax1.set_title('Завдання 1: Числове рішення')
ax1.set_xlabel('Товщина (м)')
ax1.set_ylabel('Час (с)')
ax1.set_zlabel('T (°C)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, u_analytical, cmap='plasma')
ax2.set_title('Завдання 2: Аналітичне рішення')
ax2.set_xlabel('Товщина (м)')
ax2.set_ylabel('Час (с)')
ax2.set_zlabel('T (°C)')

plt.tight_layout()
plt.show()