import sys
import numpy as np
import converter
import images
import splitter
from matplotlib import pyplot as plt
from scipy import signal

# Funcion principal
# AEPHE: Adaptative extended piecewise histogram equalisation
def AEPHE(img, N=3, alpha=None, beta=None, gamma=0,splits=None, acum_split = False, plot=True):
    # 1 : Transformar la imagen a HSI, computar el histograma del canal I.
    img_hsi = converter.RGB2HSI(img)

    # 1.5 :
    # Si me indica acum_split, tomo como cortes los intervalos en los cuales
    # se acumula el 100/N%  de los valores
    if acum_split:
        splits = splitter.get_acum_intervals(img_hsi[:,:,2], N)
    # Si todavía no hay splits, por defecto lo parto equitativamente
    if splits==None:
        splits = []
        step = int(256/N)
        end = step
        for i in range(N-1):
            splits.append(end)
            end += step
    if plot:
        # mostrar el histograma original y las particiones
        plt.subplot(2,2,1)
        histo_i = images.get_histo(img_hsi[:,:,2])
        for split in splits:
            plt.axvline(x=split,color='r')
        plt.plot(histo_i)
        plt.subplot(2,2,3)
        plt.imshow(img)

    # 2 - 5: aplciar el método en el canal I
    img_hsi[:,:,2] = AEPHE_aux(img_hsi[:,:,2], alpha, beta, gamma, splits)

    # 6 : Convertir denuevo a RGB
    img_rgb_equ = converter.HSI2RGB(img_hsi)

    if plot:
        # mostrar el histograma nuevo y las particiones
        plt.subplot(2,2,2)
        histo_i = images.get_histo(img_hsi[:,:,2])
        for split in splits:
            plt.axvline(x=split,color='r')
        plt.plot(histo_i)
        plt.subplot(2,2,4)
        plt.imshow(img_rgb_equ)
        plt.show()

    return img_rgb_equ

def AEPHE_aux(img_i, alpha, beta, gamma, splits):
    histo_i = images.get_histo(img_i)
    # Si no se pasaron alpha,beta,gamma, calcular los Mi/Mc
    if alpha==None:
        M_i = dameM_i(histo_i)
        M_i = np.amax([M_i, 0.05])
        M_c = dameM_c(img_i,histo_i)
        M_c = np.amin([M_c, 1.])
        print('Mi:',M_i)
        print('Mc:',M_c)
        alpha = M_i/(M_i+M_c)
        beta = M_c/(M_i+M_c)

    # 2 : Particionar el histograma recien computado en los splits
    parts_histo, parts_limits = splitter.custom_split_extend_histo(histo_i, splits)
    N = len(parts_limits)
    histo_target = [None]*N # array vacio para guardar los target histo
    w_k_functs = [None]*N # array vacio para guardar las funciones de peso
    histo_weights_values = [None]*N
    # 3 : Aplicar HE a cada histrograma particionado según el paper:
    for i in range(N): # para cada partición del histograma

        # IMPORTANTE:
        # parts_histo[i] esta en float, pero sus valores son en int, hay que
        # normalizarlo antes de calcular el target_histo

        # a : Crear el histograma uniforme según la función de peso
        # w_k sería la funcion peso para generar el histograma uniforme
        w_k = weight_function(parts_histo[i], parts_limits[i])
        w_k_functs[i] = w_k # guardo la funcion de pesaje
        histo_unif = np.zeros(256) # esta en float
        curr_weights = np.zeros(256)
        total = 0
        for j in range(0,256):
            part_range = range(int(parts_limits[i][0]), int(parts_limits[i][1]))
            if j in part_range: # si no es cero, quiere decir que el valor esta en
            # esta parte del histo
                histo_unif[j] = 1
                total += 1
            else:
                # el valor debe ser 0, es decir, no esta en la parte,
                # aplico weight_function
                histo_unif[j] = w_k(j)
                total += histo_unif[j]
            # calculo un array que tiene para cada nivel i, w_k_i
            # (weight_function of the piece-histogram) -> SE USA EN (4)
            curr_weights[j] = w_k(j)
        # normalizo el histograma uniforme, para que describa una distribucion
        for j in range(256):
            histo_unif[j] /= total
        # guardo una copia de los pesos para la parte i del histo
        histo_weights_values[i] = np.copy(curr_weights)
        # a.2 : calcular la matriz de suavidad D
        D  = np.zeros((255,256))
        for h in range(0,255):
            D[h][h] = -1
            D[h][h+1] = 1
        D_T = np.transpose(D) # D transpuesta
        ident = np.identity(256)
        # TODO: si vamos a fijar gamma en 0 podríamos volar una parte de acá
        # b : Resolver el sistema lineal, para hallar el target_histogram
        term_1 = np.multiply((alpha + beta), ident) + np.dot(np.multiply(gamma, D_T), D)
        term_1 = np.linalg.inv(term_1)
        # aca normalizo parts_histo[i]
        parts_histo[i] = np.divide(parts_histo[i],len(img_i)*len(img_i[0]))
        term_2 = np.multiply(alpha, parts_histo[i]) + np.multiply(beta, histo_unif)
        histo_target[i] = np.dot(term_1, term_2) # guardo el target histo

    # 4 : Juntar los histogramas una vez ecualizados, por el peso
    # ya tengo los pesos de cada parte de los histos, ahora creo uno que tenga los pesos totales
    total_weights = np.zeros(256) # ya es float
    for i in range(0,256):
        local_sum = 0.
        for n in range(0,N):
            local_sum = local_sum + histo_weights_values[n][i]
        total_weights[i] = local_sum

    # merge de histogramas
    histo_equ = np.zeros(256)
    for i in range(0,256):
        # acumulo los histos con peso
        histo_equ[i] = sum([ histo_weights_values[j][i] / total_weights[i] * histo_target[j][i]\
                for j in range(0,N)])
    # relativizar el histograma final, para que descria una distribucion
    # NOTE: !! esto de acá no debería tener que hacerse. debería dar ya una
    # distribucion
    total = sum(histo_equ)
    histo_equ /= total

    # 5 : Obtener el canal-I final, por HM
    img_i = images.HM(img_i, histo_equ)
    return img_i

# calcula en base al histograma, la funcion de peso para generar el histo_unif
# adaptado a la particion
def weight_function(piece_histo, limits):
    # esta parte esta en pag. 1014 del paper, secc 3.3
    # u_k
    first_val = int(limits[0])
    last_val = int(limits[1])
    # para hallar el valor medio del intervalo
    count = 0.
    mean = 0.
    # busco el primer y el ultimo valor de la particion
    # siempre se cumple last_val > first_val
    for i in range(first_val,last_val+1):
        count = count+1
        mean = mean+i
    # saco la media, eso va a ser u_k
    u_k = (first_val + last_val)/2.
    # sigma_k

    if count < .1:
        print("Count es cero!")
        print("Histo: ")
        print(piece_histo)
        print("\n\nFirst val: %d" % (first_val))
        print("\nEnd val: %d" % (last_val))
        sys.exit(0)
    # print("\n\nFirst val: %d" % (first_val))
    # print("\nEnd val: %d" % (last_val))

    mean = mean/count
    # sugerido por paper
    sigma_th = np.abs(128-mean)
    # cantidad de valores que cubre el intervalo
    w_d = last_val - first_val
    # tomo el maximo
    sigma_k = np.amax([w_d,sigma_th])
    return lambda i: np.exp(-((i - u_k)**2)/(2*(sigma_k**2)))


def dameM_i(histo_i):
    I_max = 255
    sigma = I_max*0.2
    M_low = 0 # ????
    phi = lambda i: np.exp(-(i-I_max)**2/(2*sigma**2))
    suma = sum([ histo_i[i] * phi(i) for i in range(256)])
    return max((1/sum(histo_i))*suma,M_low)

def dameM_c(img_i,histo_i):
    suma1 = 1/sum(histo_i)

    S = [[-1,0,1],[-2,0,2],[-1,0,1]]
    S_t = [[-1,-2,-1],[0,0,0],[1,2,1]]
    G_x = signal.fftconvolve(S,img_i)
    G_y = signal.fftconvolve(S_t,img_i)
    # G = sobel_convolution(img_i)

    R = [[] for l in range(256)]
    for x in range(len(img_i)):
        for y in range(len(img_i[0])):
            R[img_i[x][y]].append((x,y))

    suma2 = 0
    for i in range(256):
        suma3 = 0
        for (x,y) in R[i]:
            I_xy = img_i[x][y]+1
            G = np.sqrt(G_x[x][y]**2+G_y[x][y]**2)
            suma3 += G/I_xy
        suma2 += suma3

    M_up = .9
    return suma1*suma2
