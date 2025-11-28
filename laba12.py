import numpy as np

# Размер области
Nx = 50
Ny = 50

# Шаги сетки
hx = 1.0 / Nx
hy = 1.0 / Ny

# Правая часть (источники)
f = np.ones((Nx+1, Ny+1))

# Массив решения
u = np.zeros((Nx+1, Ny+1))

# Параметр релаксации (оптимальный для квадрата)
omega = 1.9

# Критерий остановки
eps = 1e-4

while True:
    max_diff = 0.0

    for j in range(1, Nx):
        for i in range(1, Ny):

            u_old = u[j, i]

            u_new = (
                (u[j+1, i] + u[j-1, i]) / hx**2 +
                (u[j, i+1] + u[j, i-1]) / hy**2 -
                f[j, i]
            ) / (2/hx**2 + 2/hy**2)

            # Релаксация
            u[j, i] = u_old + omega*(u_new - u_old)

            diff = abs(u[j, i] - u_old)
            if diff > max_diff:
                max_diff = diff

    if max_diff < eps:
        break

print("Решение найдено!")
print(u)
