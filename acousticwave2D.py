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
    
@numba.jit(parallel=True)
def v(nx, nz):
    v = np.zeros((nx,nz))
    for i in numba.prange(nx):
        for j in numba.prange(nz):
            v[i, j] = 1500
    return v

def ricker(f0, t):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

def ondas(nx,nz):
    u_anterior = np.zeros((nx,nz))
    u = np.zeros((nx,nz))
    u_posterior = np.zeros((nx,nz))
    return u_anterior, u, u_posterior

@numba.jit(parallel=True)
def rec(recx, dx, recz, dz):
    recindex = []
    for i in numba.prange(len(recx)):
        for j in numba.prange(len(recz)):
            ix = int(recx[i]/dx)
            iz = int(recz[j]/dz)
            recindex.append((ix, iz))
    return recindex

@numba.jit(parallel=True)
def marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz):
    a = (c*dt/dx)**2
    b = (c*dt/dz)**2
    if np.any(a > 1) or np.any(b > 1):
            raise ValueError("O fator de estabilidade é maior que 1. Ajuste dx ou dt.")
    for i in numba.prange(1, nx - 1):
        for j in numba.prange(1, nz-1):
            u_posterior[i,j] = 1/12 * (a[i,j]*(u[i-2,j]+u[i+2,j]-16*(u[i-1,j]+u[i+1,j])+30*u[i,j])+b[i,j]*(u[i,j-2]+u[i,j+2]-16*(u[i,j-1]+u[i,j+1])+30*u[i,j])+2*u[i,j]-u_anterior[i,j])
    return u_posterior

def marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recindex):
    isx = int(nx/2)
    isz = int(nz/2)
    sism = np.zeros((nt, len(recx)))
    plt.ion()     
    for k in numba.prange(nt):
        u[isx,isz]= u[isx,isz] + source[k]*c[isx, isz]
        u_posterior = marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz)
        u_anterior = np.copy(u)
        u = np.copy(u_posterior)

        for j, (ix, iz) in enumerate(recindex):
            sism[k, j] = u[iz, ix]
            
        if k % 200 == 0:
            plt.clf()
            M = np.max(np.abs(u.T))
            plt.imshow(u,  cmap="seismic", aspect="auto", extent=[0, (nx*dx)-1, (nz*dz)-1, 0], vmin=-M, vmax=M)
            plt.colorbar(label='Amplitude')
            plt.xlabel('x (m)')
            plt.ylabel('z (m)')
            plt.title(f'Tempo: {k * dt:.4f} s')
            plt.pause(0.05)
    plt.ioff()
    plt.show()

    return sism

def plot_sismograma(sism):
    k = np.max(np.abs(sism))
    plt.imshow(sism, cmap="seismic", aspect="auto", extent=[0, L, T, 0], vmin=-k, vmax=k)
    plt.colorbar(label='Amplitude')
    plt.title("Sismograma")
    plt.tight_layout()
    plt.show()


L = 1000
T = 1    
H = 1000
dz = 0.5               
dx = 0.5         
dt = 0.0002 
x, z, t, nx, nz, nt = parametros(L, H, T, dx, dz, dt)
c = v(nx,nz)
f0 = 30
source = ricker(f0, t)
u_anterior, u, u_posterior = ondas(nx,nz)
recx= list(range(nx)) #lista que contém a posições dos receptores de 0 até nx
recz = np.zeros(nz)
recindex = rec(recx, dx, recz, dz)
sism = marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recindex)
plot_sismograma(sism)