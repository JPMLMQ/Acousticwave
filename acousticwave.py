import numpy as np
import matplotlib.pyplot as plt

def parametros(L,T,dx,dt):
    x = np.arange(0,L+dx, dx)
    t = np.arange(0,T+dt, dt)
    nx = len(x) 
    nt = len(t)
    return x,t,nx,nt

#caso c seja uma função
#def c(x):
    #return 2x-3

def ricker(f0,t):
    pi = np.pi
    td  = t - 2*np.sqrt(pi)/f0
    fcd = f0/(np.sqrt(pi)*3) 
    source = (1 - 2*pi*(pi*fcd*td)*(pi*fcd*td))*np.exp(-pi*(pi*fcd*td)*(pi*fcd*td))
    return source

def ondas(nx):
    u_anterior = np.zeros(nx)
    u = np.zeros(nx)
    u_posterior = np.zeros(nx)
    return u_anterior,u,u_posterior

def rec(recx,dx):
    recindex = np.zeros(len(recx))
    for i in range(len(recx)):
        recindex[i] = int(recx[i] / dx)
    return recindex 

def marcha_no_tempo(u_anterior, u, u_posterior, source, c, dt, dx, nt, nx, recx, recindex):
    isx = int(nx / 2)
    sism = np.zeros((len(recx), nt))  
    R = (c*dt/dx)**2
    if R>1:
        raise ValueError("O fator de estabilidade é maior que 1. Ajuste dx ou dt.")
    for k in range(nt):
        u[isx] = u[isx] + source[k]*c
        for i in range(1, nx - 1):
            u_posterior[i] = R*u[i-1] - 2*(R-1)*u[i] + R*u[i+1] - u_anterior[i]

        u_anterior = np.copy(u)
        u = np.copy(u_posterior)

        for j, idx in enumerate(recindex):
            sism[j, k] = u[int(idx)]

    return sism

def animacao(u_anterior, u, u_posterior, source, c, dt, dx,nt,nx,recx,recindex):
    isx = int(nx/2)
    plt.ion()
    for k in range(nt):
        u[isx] = u[isx] + source[k]*c
        for i in range(1,nx-1):
            R = (c*dt/dx)**2
            if R > 1:
                raise ValueError("O fator de estabilidade é maior que 1. Ajuste dx ou dt.")
            else:
                u_posterior[i] = R*u[i-1] - 2*(R-1)*u[i] + R*u[i+1] - u_anterior[i]
        u_anterior = np.copy(u)
        u = np.copy(u_posterior)

        if k % 200 == 0:
            plt.clf()  
            plt.plot(x, u)  
            plt.xlabel('x (m)')
            plt.ylabel('Amplitude')
            plt.title(f'Tempo: {k * dt:.4f} s')
            plt.grid(True)
            plt.pause(0.05)
    plt.ioff()
    plt.show()
  
def plot_receptor(t,sism):
    plt.plot(t, sism[0, :])
    plt.title("Sism (x=80 m)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def plot_sismograma(sism):
    k = np.max(np.abs(sism))
    plt.imshow(sism, cmap="seismic", aspect="auto", extent=[0, L, T, 0], vmin=-k, vmax= k)
    plt.colorbar(label='Amplitude')
    plt.title("Sismograma")
    plt.tight_layout()
    plt.show()


L = 1000
T = 1                  
dx = 0.5         
dt = 0.0002 
c = 1500 
#se c for uma função
#c = c(1500)  
x,t,nx,nt = parametros(L,T,dx,dt)
f0 = 30
source = ricker(f0,t)
plt.plot(t,source)
plt.show()
u_anterior, u, u_posterior = ondas(nx)
recx = [800]
recindex = rec(recx, dx)
sism = marcha_no_tempo(u_anterior, u, u_posterior, source, c, dt, dx, nt, nx, recx, recindex)
animacao(u_anterior, u, u_posterior, source, c, dt, dx,nt,nx,recx,recindex)
plot_receptor(t,sism)
plot_sismograma(sism)