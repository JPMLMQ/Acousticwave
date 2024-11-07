import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker

L = 1.0          
T = 0.1                  
dx = 0.05          
dt = 0.01      

nx = int(L / dx)  
nt = int(T / dt) 
f0 = 30
n = 2
sigma = n / (2 * np.pi * f0)

def ricker(t, sigma):
    return (1 -(t/sigma)**2)*np.exp(-(t)**2/(2*sigma**2))

x = np.linspace(0, L, nx)

def c(x):
    return 2 + x/3

u = np.zeros(nx)
u_anterior = np.zeros(nx)
u_posterior = np.zeros(nx)

posição = int(nx / 2)
for i in range(nx):
    u[i] = ricker(x[i] - x[posição], sigma)

soluções = np.zeros((nx, nt))

for t in range(nt):
    for i in range(nx-1):
        R = (c(x[i])*dt/dx)**2
        u_posterior[i] = R*u[i-1] - 2*(R-1)*u[i] + R*u[i+1] - u_anterior[i]

    u_anterior = u.copy()
    u = u_posterior.copy()
    
    soluções[:, t] = u

snapshot = int(nt/2)


plt.plot(x, soluções[:,snapshot], label=f"tempo(t={snapshot}s)")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()