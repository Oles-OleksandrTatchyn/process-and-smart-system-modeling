import numpy as np
import matplotlib.pyplot as plt

# Функція для Runge-Kutta 4 порядку
def runge_kutta_system(f, y0, t0, T, h):
    t_values = np.arange(t0, T+h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)

        y_values[i] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t_values, y_values

# Завдання 1: екосистема "жертва-хижак"
# Параметри
N = 20
a11 = 0.01 * N  # 0.2
a12 = 0.0001 * N # 0.002
a21 = 0.0001 * N # 0.002
a22 = 0.04 * N   # 0.8

x0 = 1000 - 10*N  # 800
y0 = 700 - 10*N   # 500
t0 = 0
h = 0.1
T1 = 150

# Опис системи Лотки-Вольтери
def lotka_volterra(t, vars):
    x, y = vars
    dxdt = a11 * x - a12 * x * y
    dydt = -a22 * y + a21 * x * y
    return np.array([dxdt, dydt])

# Розрахунок
t1, result1 = runge_kutta_system(lotka_volterra, [x0, y0], t0, T1, h)
x, y = result1[:,0], result1[:,1]

# Побудова графіків
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(t1, x, label='Жертви (Зайці)')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість')
plt.title('Кількість зайців x(t)')
plt.legend()

plt.subplot(1,3,2)
plt.plot(t1, y, label='Хижаки (Вовки)', color='r')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість')
plt.title('Кількість вовків y(t)')
plt.legend()

plt.subplot(1,3,3)
plt.plot(x, y)
plt.xlabel('Жертви (Зайці)')
plt.ylabel('Хижаки (Вовки)')
plt.title('Фазова траєкторія y(x)')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

# Завдання 2: розповсюдження епідемії
H = 1000 - N  # 980
beta = 25 - N # 5
gamma = N     # 20

x0_epid = 900 - N  # 880
y0_epid = 90 - N   # 70
z0_epid = H - x0_epid - y0_epid # 30

T2 = 40

# Опис системи епідемії
def epidemic(t, vars):
    x, y, z = vars
    dxdt = -beta * x * y / H
    dydt = beta * x * y / H - y / gamma
    dzdt = y / gamma
    return np.array([dxdt, dydt, dzdt])

# Розрахунок
t2, result2 = runge_kutta_system(epidemic, [x0_epid, y0_epid, z0_epid], t0, T2, h)
x_e, y_e, z_e = result2[:,0], result2[:,1], result2[:,2]

# Побудова графіків
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(t2, x_e, label='Здорові')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість людей')
plt.title('Здорові x(t)')
plt.legend()

plt.subplot(1,3,2)
plt.plot(t2, y_e, label='Хворі', color='r')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість людей')
plt.title('Хворі y(t)')
plt.legend()

plt.subplot(1,3,3)
plt.plot(t2, z_e, label='Перехворілі', color='g')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість людей')
plt.title('Перехворілі z(t)')
plt.legend()

plt.tight_layout()
plt.show()
