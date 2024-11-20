import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker

L = 100          
T = 1                  
dx = 1          
dt = 0.01      

nx = int(L / dx)  
nt = int(T / dt) 
f0 = 30
n = 2

def ricker(t, f0):
    return (1 - 2 * (np.pi * f0* t)**2)*np.exp(-(np.pi * f0* t)**2)

x = np.linspace(0, L, nx)
t = np.linspace(0,T, nt)

def c(x):
    return 2000

u = np.zeros(nx)
u_anterior = np.zeros(nx)
u_posterior = np.zeros(nx)

F = np.zeros(nt)
tlag = 0.04

for i in range(nt):
    F[i] = ricker(t[i]- tlag, f0)

solution = np.zeros((nx, nt))
isx = int(nx/2)

for t in range(nt):
    for i in range(nx-1):
        R = (c(x[i])*dt/dx)**2
        u_posterior[isx] = F[t]
        u_posterior[i] += R*u[i-1] - 2*(R-1)*u[i] + R*u[i+1] - u_anterior[i]
        
    u_anterior = u.copy()
    u = u_posterior.copy()
    
    solution[:, t] = u

snapshot = 0

plt.plot(x, solution[:,snapshot], label=f"tempo(t={snapshot}s)")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()

k = np.max(np.abs(solution))

plt.imshow(solution, cmap="seismic", aspect="auto", extent=[0, L, T, 0], vmin=-k, vmax=k)
plt.colorbar(label='Amplitude')
plt.title("Sismograma")
plt.tight_layout()
plt.show()

