import numpy as np
import matplotlib.pyplot as plt

# параметры
N = 100
h = 1.0 / N

x = np.linspace(0, 1, N+1)

# индекс дельта-функции
x0 = 0.5
j0 = int(x0 / h)

# правая часть
f = np.zeros(N+1)
f[j0] = 1.0 / h   # дискретизация дельта-функции

# создаем матрицу разностной схемы
A = np.zeros((N-1, N-1))
b = np.zeros(N-1)

for i in range(N-1):
    A[i, i] = -2
    if i > 0:
        A[i, i-1] = 1
    if i < N-2:
        A[i, i+1] = 1

    b[i] = -h*h*f[i+1]

# решаем систему
u_inner = np.linalg.solve(A, b)

# формируем полное решение
u = np.zeros(N+1)
u[1:N] = u_inner

# точное решение
u_exact = np.where(x < x0, x, 1-x)

# график
plt.plot(x, u, label="Численное решение")
plt.plot(x, u_exact, '--', label="Точное решение")
plt.legend()
plt.grid()
plt.title("Лабораторная 13: решение с дельта-функцией")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.show()
