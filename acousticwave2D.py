import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
import pandas as pd

# def v(nx, nz, v1=1500, v2=2000, v3=3000):
#     vp = np.zeros([nz,nx])
#     vp[0:int(nz/4),:]= v1
#     vp[int(nz/4):int(nz/2),:] = v2
#     vp[int(nz/2):nz,:] = v3
#     return vp

def ler_modelo(caminho_arquivo, shape):
    vp = np.fromfile(caminho_arquivo, dtype=np.float32)
    vp = vp.reshape(shape)
    print(f"Modelo de velocidade carregado de: {caminho_arquivo}")
    return vp

def plot_modelo(vp):
    plt.imshow(vp.T, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.show()

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

def borda (nx,nz,fator = 0.015, N = 100):
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
    sism_shot = []
    fig, ax = plt.subplots(figsize=(10, 10))  
    for i_shot, (sx, sz) in enumerate(zip(isx, isz)):
        u_anterior.fill(0)  
        u.fill(0)
        u_posterior.fill(0)
        sism_atual = np.zeros((nt, nx))

        for k in range(nt):
            u[sz,sx]= u[sz,sx] + source[k]*(dt*c[sz, sx])**2
            u_posterior = marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz) 
            u_posterior *= A
            u_anterior = np.copy(u) 
            u_anterior *= A
            u = np.copy(u_posterior)

            sism_atual[k, recx] = u[recz, recx]
            sism[k, recx] += u[recz, recx]
                
            if (k%100 == 0):    
                ax.cla()
                ax.imshow(u)
                plt.pause(0.1)

        sism_shot.append(sism_atual)
    return sism,sism_shot

def plot_sismograma(sism):
    perc = np.percentile(sism,99)
    plt.imshow(sism,aspect='auto',cmap='gray',vmin=-perc,vmax=perc)
    plt.colorbar(label='Amplitude')
    plt.title("Sismograma")
    plt.show()

def plot_shot(sism_shot):
    sism_shot.reverse()
    for i in range(len(sism_shot)):
        perc = np.percentile(sism_shot[i], 99)
        plt.imshow(sism_shot[i], aspect='auto', cmap='gray', vmin=-perc, vmax=perc)
        plt.colorbar(label='Amplitude')
        plt.title(" shot %s"%i)
        plt.show()

def salvar_sismograma(sism):
    sism.tofile(f'D:/GitHub/Acousticwave/sismograma_{sism.shape}.bin')   
    print(f"Sismograma salvo em: D:/GitHub/Acousticwave/sismograma_{sism.shape}.bin")


receiverTable = pd.read_csv('d:/GitHub/Geofisica/receivers.csv')
sourceTable = pd.read_csv('d:/GitHub/Geofisica/sources.csv')
rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()

L  = 10000
H = 3000
T = 2 
dt = 0.001 
N = 100
nx = 383
nz = 141
nx_abc = nx + 2*N
nz_abc = nz + 2*N
dx = L/nx
dz = H/nz
f0 = 60
x = np.arange(0,L+dx,dx)
z = np.arange(0,H+dz,dz)
t = np.arange(0,T+dt,dt)
nt = len(t)
c = ler_modelo('D:/GitHub/Acousticwave/marmousi_vp_383x141.bin', (383, 141))
c_expand = expand_vp(c,nx_abc,nz_abc, N)
source = ricker(f0, t)

#critérios de dispersão e estabilidade
vp_min= np.min(c_expand)
vp_max = np.max(c_expand)
lambda_min = vp_min / f0
dx_lim = lambda_min / 10
dt_lim = dx_lim / (np.sqrt(2) * vp_max)
if (dt>=dt_lim and dx>=dx_lim):
    print("Condições de estabilidade e dispersão satisfeitas")
else:
    print("Condições de estabilidade e dispersão não satisfeitas")
    print("dt_critical = %f dt = %f" %(dt_lim,dt))
    print("dx_critical = %f dx = %f" %(dx_lim,dx))
    print("fcut = %f " %(f0))


u_anterior, u, u_posterior = ondas(nx,nz)
recx= range(nx)
recz = N + 10
A = borda(nx, nz, fator=0.015, N = 100)
sism,sism_shot = marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recz,dt, A, shot_x, shot_z, dx, dz)
plot_sismograma(sism)
plot_shot(sism_shot)
salvar_sismograma(sism)