import numpy as np
import matplotlib.pyplot as plt
#from IPython import display

dx = 0.5
dt = 0.0002
T  = 1
L  = 1000

x = np.arange(0,L+dx,dx)
t = np.arange(0,T+dt,dt)

Nx = len(x)
Nt = len(t)

vp = 1500

#Source
freq = 30
pi = np.pi
td  = t - 2*np.sqrt(pi)/freq
fcd = freq/(np.sqrt(pi)*3) 
source = (1 - 2*pi*(pi*fcd*td)*(pi*fcd*td))*np.exp(-pi*(pi*fcd*td)*(pi*fcd*td))

# Check wavelet
plt.figure(figsize=(16,4))
plt.plot(t,source)
plt.xlabel('tempo')
plt.show()

Up = np.zeros(Nx)
Uc = np.zeros(Nx)
Uf = np.zeros(Nx)

sx = int(Nx/2)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for k in range(Nt):
  Uc[sx] = Uc[sx] + source[k]*vp
  for i in range(1,Nx-1):
    Uf[i] = (vp*dt/dx)*(vp*dt/dx)* (Uc[i+1] - 2*Uc[i] + Uc[i-1]) + 2*Uc[i] - Up[i]

  Up = np.copy(Uc)
  Uc = np.copy(Uf)

  if k%200==0:
    plt.cla()
    plt.plot(x,Uc,'.')
    plt.xlabel('x (m) - time %2.4f' %(k*dt))
    plt.ylabel('Amplitude')
    plt.show()
