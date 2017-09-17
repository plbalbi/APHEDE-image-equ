import numpy as np
from scipy import ndimage
import converter
import images
# AEPHE: Adaptative extended piecewise histogram equalisation

# Funcion principal
def AEPHE(img, N):
    # 1 : Transformar la imagen a HSI, computar el histograma del canal I.
    img_hsi = converter.RGB2HSI(img)
    histo_i = images.get_histo(img_hsi[:,:,2])
    # 2 : Particionar el histograma recien computado en N-partes, y a cada una de ellas,
    # Como primer approach, partimos en N partes de igual tamaño, disjuntas
    # extenderlas según (6).
    parts_histo = split_extend_histo(histo_i, N)
        # previo_a_3 : TODO: Computar M_i y M_c según el paper, los cuales son los parámetros alpha y beta
    # 3 : Aplicar HE a cada histrograma particionado según el paper:
    for i in range(0,N): # para cada partición del histograma
        # a : Crear el histograma uniforme según la función de peso
        # b : Resolver el sistema lineal, para hallar el target_histogram
    # 4 : Juntar los histogramas una vez ecualizados, por el peso
    # 5 : Obtener el canal-I final, por HM
    # 6 : Convertir denuevo a RGB
    img_rgb_equ = converter.HSI2RGB(img_hsi)
    return img_rgb_equ


# Parte el histograma histo en N partes, y lo extiende según (6) en el paper
def split_extend_histo(histo, N):
    histo_parts = []
    step = np.floor(256./float(N))
    init = 0
    end = step
    for i in range(0,N):
        init = step*i+1
        if i == 0:
            step = step-1 # para que empieza en 0 la primera vez
        end = i*step
        if i == N-1:
            end = 255 # el end de la ultima particion es directo 255
        # inicio una de las partes en 0's
        histo_parts[i] = np.zeros(256)
        for j in range(init, end+1):
            # copio la parte que se encuentra dentro del rango
            histo_parts[i][j] = histo[j]
    return histo_parts

