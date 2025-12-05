import numpy as np
import matplotlib.pyplot as plt

# параметры задачи
l = 1
T = 1.0

h = 0.05
tau = 0.0005

Nz = int(2*l / h)
Nt = int(T / tau)

z = np.linspace(-l, l, Nz+1)
t = np.linspace(0, T, Nt+1)

# точное решение
def exact(z, t):
    return t**2 * np.exp(-z**2)

# правая часть f(z,t)
def f(z, t):
    return 2*t*np.exp(-z**2) - (4*z**2 - 2)*(t**2)*np.exp(-z**2)

# массив численного решения
u = np.zeros((Nt+1, Nz+1))

# начальное условие u(z,0) = 0
u[0, :] = 0

# явная схема
for n in range(Nt):
    for i in range(1, Nz):
        u[n+1, i] = u[n, i] + tau * (
            (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / h**2 +
            f(z[i], t[n])
        )
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# точное решение в конечный момент времени
u_exact = exact(z, T)

# график
plt.plot(z, u[-1], label="Явная схема")
plt.plot(z, u_exact, label="Точное решение", linestyle="dashed")
plt.legend()
plt.title("Сравнение явной схемы и точного решения")
plt.xlabel("z")
plt.ylabel("u(z,T)")
plt.grid()
plt.show()
