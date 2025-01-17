import numpy as np
import matplotlib.pyplot as plt

def ler_sismograma(caminho_arquivo, shape):
    sism = np.fromfile(caminho_arquivo, dtype=np.float64)  
    sism = sism.reshape(shape)
    print(f"Sismograma carregado de: {caminho_arquivo}")
    return sism

def plot_sismograma(sism):
    perc = np.percentile(sism,99)
    plt.imshow(sism,aspect='auto',cmap='gray',vmin=-perc,vmax=perc)
    plt.colorbar(label='Amplitude')
    plt.title("Sismograma")
    plt.show()

def ler_modelo(caminho_arquivo, shape):
    vp = np.fromfile(caminho_arquivo, dtype=np.float32)
    vp = vp.reshape(shape)
    print(f"Modelo de velocidade carregado de: {caminho_arquivo}")
    return vp

def plot_modelo(vp):
    plt.imshow(vp.T, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.show()

# sism = ler_sismograma('D:/GitHub/Acousticwave/sismograma_(2001, 1001).bin', (2001, 1001))
# plot_sismograma(sism)

vp = ler_modelo('D:/GitHub/Acousticwave/marmousi_vp_383x141.bin', (383, 141))
plot_modelo(vp)