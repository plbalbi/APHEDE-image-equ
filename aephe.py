import numpy as np
from scipy import ndimage
import converter
import images
# AEPHE: Adaptative extended piecewise histogram equalisation

# Funcion principal
# por defecto les paso un tercio, para que no explote si me olvido de pasarle algo
def AEPHE(img, N, alpha=1./3., beta=1./3., gamma=1./3.):
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

        # IMPORTANTE:
        # parts_histo[i] esta en float, pero sus valores son en int, hay que
        # normalizarlo antes de calcular el target_histo

        # a : Crear el histograma uniforme según la función de peso
        # w_k sería la funcion peso para generar el histograma uniforme
        w_k = weight_function(parts_histo[i])
        histo_unif = np.zeros(256) # esta en float
        for j in range(0,256):
            if parts_histo[i][j]>0.1: # si no es cero, quiere decir que el valor esta en
            # esta parte del histo
                histo_unif[j] = 1
            else:
                # el valor debe ser 0, es decir, no esta en la parte,
                # aplico weight_function
                histo_unif[j] = w_k(j)
                
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

# calcula en base al histograma, la funcion de peso para generar el histo_unif
# adaptado a la particion
def weight_function(piece_histo):
    # esta parte esta en pag. 1014 del paper, secc 3.3
    # u_k
    first_val = 0
    first_found = False
    last_val = 0
    # para hallar el valor medio del intervalo
    count = 0
    mean = 0.
    # busco el primer y el ultimo valor de la particion
    # siempre se cumple last_val > first_val
    for i in range(0,256):
        # como se que son float, pero en relidad enteros, evaluo mayor a 0.1
        if piece_histo[i] > 0.1:
            if not first_found:
                first_val = i
                first_found = True
            else:
                last_val = i
            count = count+1
            mean = mean+i
    # saco la media, eso va a ser u_k
    u_k = (first_val + last_val)/2.
    # sigma_k
    mean = mean/float(count)
    # sugerido por paper
    sigma_th = np.abs(128-mean)
    # cantidad de valores que cubre el intervalo
    w_d = last_val - first_val
    # tomo el maximo
    sigma_k = np.amax([w_d,sigma_th])
    return lambda i: np.exp(-((i - u_k)**2)/(2*(sigma_k**2)))
