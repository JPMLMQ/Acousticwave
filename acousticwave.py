import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker

L = 1000
T = 1                  
dx = 0.5         
dt = 0.0002    

x = np.arange(0,L+dx, dx)
t = np.arange(0,T+dt, dt)

nx = len(x) 
nt = len(t)
f0 = 30
n = 2
c = 1500

def ricker(t, f0):
    return (1 - 2 * (np.pi * f0* t)**2)*np.exp(-(np.pi * f0* t)**2)

#def c(x):
    return 100

u = np.zeros(nx)
u_anterior = np.zeros(nx)
u_posterior = np.zeros(nx)

F = np.zeros(nt)
tlag = 0.04

for i in range(nt):
    F[i] = ricker(t[i]- tlag, f0)

plt.plot(t, F)
plt.show()

solution = np.zeros((nx, nt))
isx = int(nx/2)
recx = [800]
recindex = np.zeros(len(recx))
for i in range(len(recx)):
    recindex[i] = int(recx[i] / dx)

sism = np.zeros((len(recx),nt))

for t in range(nt):
    for i in range(nx-1):
        R = (c*dt/dx)**2
        u_posterior[isx] = F[t]
        u_posterior[i] += R*u[i-1] - 2*(R-1)*u[i] + R*u[i+1] - u_anterior[i]
        
    for j, k in range(len(recindex)):
        sism[j, t] = u[recindex[k]]
    
    u_anterior = u.copy()
    u = u_posterior.copy()
    
    solution[:, t] = u

    if t%200==0:
        plt.plot(x, solution[:,t], label=f"tempo(t={t}s)")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.grid(True)
        plt.show()


plt.plot(t, sism)
plt.title("Sismograma no Receptor (x=80 m)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

k = np.max(np.abs(solution))

plt.imshow(solution, cmap="seismic", aspect="auto", extent=[0, L, T, 0], vmin=-k, vmax=k)
plt.colorbar(label='Amplitude')
plt.title("Sismograma")
plt.tight_layout()
plt.show()
