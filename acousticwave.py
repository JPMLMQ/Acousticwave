import numpy as np
import matplotlib.pyplot as plt

L = 1000
T = 1                  
dx = 0.5         
dt = 0.0002    

x = np.arange(0,L+dx, dx)
t = np.arange(0,T+dt, dt)

nx = len(x) 
nt = len(t)
c = 1500
#caso c seja uma função
#def c(x):
    #return 100

#ricker
f0 = 30
pi = np.pi
td  = t - 2*np.sqrt(pi)/f0
fcd = f0/(np.sqrt(pi)*3) 
source = (1 - 2*pi*(pi*fcd*td)*(pi*fcd*td))*np.exp(-pi*(pi*fcd*td)*(pi*fcd*td))

plt.plot(t,source)
plt.xlabel('tempo')
plt.show()


u_anterior = np.zeros(nx)
u = np.zeros(nx)
u_posterior = np.zeros(nx)


isx = int(nx/2)
recx = [800]
recindex = np.zeros(len(recx))
for i in range(len(recx)):
    recindex[i] = int(recx[i] / dx)

sism = np.zeros((len(recx),nt))

for k in range(nt):
    u[isx] = u[isx] + source[k]*c
    for i in range(1,nx-1):
        R = (c*dt/dx)**2
        u_posterior[i] = R*u[i-1] - 2*(R-1)*u[i] + R*u[i+1] - u_anterior[i]

    u_anterior = np.copy(u)
    u = np.copy(u_posterior)
        
    for j, idx in enumerate(recindex):
        sism[j, k] = u[int(idx)]

    if k%200==0:
        plt.plot(x, u)
        plt.xlabel('x (m) - time %2.4f' %(k*dt))
        plt.ylabel('Amplitude')
        plt.show()
    

plt.plot(t, sism[0, :])
plt.title("Sism (x=80 m)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

k = np.max(np.abs(sism))

plt.imshow(sism, cmap="seismic", aspect="auto", extent=[0, L, T, 0], vmin=-k, vmax= k)
plt.colorbar(label='Amplitude')
plt.title("Sismograma")
plt.tight_layout()
plt.show()
