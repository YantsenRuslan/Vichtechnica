import numpy as np
import matplotlib.pyplot as plt

# параметры сетки
Nx = 50
Nt = 200
L = 1.0
T = 1.0

hx = L / Nx
ht = T / Nt

x = np.linspace(0, L, Nx+1)
t = np.linspace(0, T, Nt+1)

# позиция измерения
x0 = 0.5
i0 = int(x0 / hx)

# задаём "истинные" измерения (для лабораторной)
g = np.sin(np.pi * t)

# начальное приближение f(x)
f = np.zeros(Nx+1)

# шаг градиентного спуска
alpha = 0.5

# функции
def solve_forward(f):
    u = np.zeros((Nt+1, Nx+1))
    for n in range(Nt):
        for i in range(1, Nx):
            u[n+1, i] = u[n, i] + ht * (
                (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / hx**2
                + f[i]
            )
    return u

def solve_adjoint(u):
    psi = np.zeros((Nt+1, Nx+1))
    for n in reversed(range(Nt)):
        psi[n, i0] += u[n, i0] - g[n]    
        for i in range(1, Nx):
            psi[n, i] += psi[n+1, i] + ht * (
                (psi[n+1, i+1] - 2*psi[n+1, i] + psi[n+1, i-1]) / hx**2
            )
    return psi

def compute_gradient(psi):
    return -np.trapz(psi, dx=ht, axis=0)

# основной цикл оптимизации
for k in range(10):
    u = solve_forward(f)
    psi = solve_adjoint(u)
    grad = compute_gradient(psi)

    f -= alpha * grad

print("Готово! f(x) определена.")

plt.plot(x, f)
plt.title("Восстановленная функция f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()
