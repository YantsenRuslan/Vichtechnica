import numpy as np
import matplotlib.pyplot as plt

L = 1.0
a = 1.0
N = 100
dx = L / N

T = 2.0
dt = 0.005

c = a * dt / dx
print("Число Куранта c =", c)
if abs(c) >= 1:
    print("!!! СХЕМА НЕУСТОЙЧИВА !!! уменьшите dt")
    exit()

x = np.linspace(0, L, N+1)

u_prev = np.zeros(N+1)
u_curr = np.zeros(N+1)
u_next = np.zeros(N+1)

u_prev = np.sin(np.pi * x)
u_curr[:] = u_prev[:]

for j in range(1, N):
    u_xx = (u_prev[j+1] - 2*u_prev[j] + u_prev[j-1]) / dx**2
    u_curr[j] = u_prev[j] + 0.5 * dt*dt * a*a * u_xx

u_curr[0] = 0
u_curr[-1] = 0

steps = int(T / dt)

plt.figure(figsize=(8,4))

for step in range(steps):
    for j in range(1, N):
        u_next[j] = (2*(1 - c*c)*u_curr[j]
                     + c*c*(u_curr[j+1] + u_curr[j-1])
                     - u_prev[j])

    u_next[0] = 0
    u_next[-1] = 0

    u_prev[:] = u_curr[:]
    u_curr[:] = u_next[:]

    if step % 40 == 0:
        plt.clf()
        plt.plot(x, u_curr)
        plt.ylim(-1.1, 1.1)
        plt.title(f"t = {step*dt:.3f}")
        plt.pause(0.01)

plt.show()
