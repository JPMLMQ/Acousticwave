import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
import pandas as pd

#mudanças para geometria externa

def parametros(L, H, T, dx, dz, dt, N):
    nx = int(L/dx) + 1
    nz = int(H/dz) + 1
    nx_abc = nx + 2*N
    nz_abc = nz + 2*N
    nt = int(T/dt) + 1
    x = np.linspace(0, L, nx)
    z = np.linspace(0, H, nz)
    t = np.linspace(0, T, nt)
    return x, z, t, nx, nz, nx_abc, nz_abc, nt
    
def v(nx, nz, v1=1500, v2=2000, v3=3000):
    vp = np.zeros([nz,nx])
    vp[0:int(nz/4),:]= v1
    vp[int(nz/4):int(nz/2),:] = v2
    vp[int(nz/2):nz,:] = v3
    return vp

def expand_vp(v,nx_abc,nz_abc, N):
    v_expand = np.zeros((nz_abc, nx_abc))
    v_expand[N:nz_abc-N, N:nx_abc - N] = v
    v_expand[0:N,N: nx_abc - N]= v[0, :]
    v_expand[nz_abc - N : nz_abc, N: nx_abc - N]= v[-1, :]
    v_expand[N:nz_abc-N, 0:N] = v[:, 0][:, np.newaxis]
    v_expand[N:nz_abc-N, nx_abc-N:nx_abc] = v[:, -1][:, np.newaxis]
    v_expand[0:N, 0:N] = v[0, 0]  
    v_expand[0:N, nx_abc-N:nx_abc] = v[0, -1] 
    v_expand[nz_abc-N:nz_abc, 0:N] = v[-1, 0]  
    v_expand[nz_abc-N:nz_abc, nx_abc-N:nx_abc] = v[-1, -1]

    return v_expand


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

def borda (nx,nz,fator = 0.015, N = 50):
    A = np.ones((nz, nx))  
    for i in range(nx):
        for j in range(nz):
            if i <= N: 
                A[j, i] *= np.exp(-((fator * (N - i)) ** 2))
            elif i >= nx - N:  
                A[j, i] *= np.exp(-((fator * (i - (nx - N))) ** 2))

            if j < N:  
                A[j, i] *= np.exp(-((fator * (N - j)) ** 2))
            elif j >= nz - N:  
                A[j, i] *= np.exp(-((fator * (j - (nz - N))) ** 2))

    return A

@numba.jit(parallel=True, nopython=True)
def marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz):
    if np.any(((c*dt/dx)**2)> 1) or np.any(((c*dt/dz)**2)> 1):
            raise ValueError("O fator de estabilidade é maior que 1. Ajuste dx, dz ou dt.")
    for i in numba.prange(2, nx - 3):
        for j in numba.prange(2, nz - 3):
            pxx = (-u[j, i+2] + 16*u[j, i+1] - 30*u[j, i] + 16*u[j, i-1] - u[j, i-2]) / (12 * dx * dx)
            pzz = (-u[j+2, i] + 16*u[j+1, i] - 30*u[j, i] + 16*u[j-1, i] - u[j-2, i]) / (12 * dz * dz)
            u_posterior[j, i] = (c[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * u[j, i] - u_anterior[j, i]
    return u_posterior

def marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recz,dt, A, shot_x, shot_z, dx, dz):
    isx = np.round(shot_x / dx).astype(int)
    isz = np.round(shot_z / dz).astype(int)
    sism = np.zeros((nt, nx)) 
    fig, ax = plt.subplots(figsize=(10, 10))  
    for k in range(nt):
        for sx, sz in zip(isx, isz):
            u[sz,sx]= u[sz,sx] + source[k]*(dt*c[sz, sx])**2
            u_posterior = marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz) 
            u_posterior *= A
            u_anterior = np.copy(u) 
            u_anterior *= A
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


receiverTable = pd.read_csv('d:/GitHub/Geofisica/receivers.csv')
sourceTable = pd.read_csv('d:/GitHub/Geofisica/sources.csv')
rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()
L = len(rec_x)
H = len(rec_z)
dx = 50
dz = 50
nx = int(L/dx) + 1
nz = int(H/dz) + 1
T = 2         
dt = 0.001 
N = 50
x, z, t, nx, nz, nx_abc, nz_abc, nt = parametros(L, H, T, dx, dz, dt, N)
c = v(nx,nz)
c_expand = expand_vp(c,nx_abc,nz_abc, N)
# plt.figure()
# plt.imshow(c_expand,aspect='equal')
# plt.show()
f0 = 60
source = ricker(f0, t)
u_anterior, u, u_posterior = ondas(nx,nz)
recx= range(nx)
recz = N + 10
A = borda(nx, nz, fator=0.015, N = 50)
sism = marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recz,dt, A, shot_x, shot_z, dx, dz)
plot_sismograma(sism)



