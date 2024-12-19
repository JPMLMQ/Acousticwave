import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit

def parametros(L, H, T, dx, dz, dt):
    nx = int(L/dx) + 1
    nz = int(H/dz) + 1
    nt = int(T/dt) + 1
    x = np.linspace(0, L, nx)
    z = np.linspace(0, H, nz)
    t = np.linspace(0, T, nt)
    return x, z, t, nx, nz, nt
    
def v(nx, nz):
    vp = np.zeros([nz,nx])
    vp[:,:]= 1500
    # vp[0:int(nz/4),:]= 1500
    # vp[int(nz/4):int(nz/2),:] = 2000
    # vp[int(nz/2):nz,:] = 3000
    return vp

def ricker(f0, t):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

def ondas(nx,nz):
    u_anterior = np.zeros((nz,nx))
    u = np.zeros((nz,nx))
    u_posterior = np.zeros((nz,nx))
    return u_anterior, u, u_posterior

@numba.jit(parallel=True, nopython=True)
def marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz):
    if np.any(((c*dt/dx)**2)> 1) or np.any(((c*dt/dz)**2)> 1):
            raise ValueError("O fator de estabilidade é maior que 1. Ajuste dx ou dt.")
    for i in numba.prange(2, nx - 3):
        for j in numba.prange(2, nz - 3):
            pxx = (-u[j, i+2] + 16*u[j, i+1] - 30*u[j, i] + 16*u[j, i-1] - u[j, i-2]) / (12 * dx * dx)
            pzz = (-u[j+2, i] + 16*u[j+1, i] - 30*u[j, i] + 16*u[j-1, i] - u[j-2, i]) / (12 * dz * dz)
            u_posterior[j, i] = (c[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * u[j, i] - u_anterior[j, i]
    return u_posterior

def marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recz,dt):
    isx = int(nx/2)
    isz = 0
    sism = np.zeros((nt, nx)) 
    fig, ax = plt.subplots(figsize=(10, 10))  
    for k in range(nt):
        u[isz,isx]= u[isz,isx] + source[k]*(dt*c[isz, isx])**2
        u_posterior = marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz)
        u_anterior = np.copy(u)
        u = np.copy(u_posterior)

        sism[k, recx] = u[recz, recx]
            
        if (k%100 == 0):    
            ax.cla()
            ax.imshow(u)
            plt.pause(0.1)
            

    return sism

def plot_sismograma(sism):
    perc = np.percentile(sism,99)
    plt.imshow(sism,aspect='auto',cmap='gray',vmin=-perc,vmax=perc)
    plt.colorbar(label='Amplitude')
    plt.title("Sismograma")
    plt.show()


L = 10000
T = 2    
H = 3000
dz = 5               
dx = 5         
dt = 0.001 
x, z, t, nx, nz, nt = parametros(L, H, T, dx, dz, dt)
c = v(nx,nz)
f0 = 60
source = ricker(f0, t)
u_anterior, u, u_posterior = ondas(nx,nz)
recx= range(nx)
recz = 10
sism = marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recz, dt)
plot_sismograma(sism)