import sys
import numpy as np
import converter
import images
from matplotlib import pyplot as plt
from scipy import signal
# AEPHE: Adaptative extended piecewise histogram equalisation

# Funcion principal
# por defecto les paso un tercio, para que no explote si me olvido de pasarle algo
def AEPHE(img, N, alpha=1./3., beta=1./3., gamma=0):
    # 1 : Transformar la imagen a HSI, computar el histograma del canal I.
    img_hsi = converter.RGB2HSI(img)
    histo_i = images.get_histo(img_hsi[:,:,2])
    num_pixels = len(img)*len(img[0]) # cantidad de pixels
    # 2 : Particionar el histograma recien computado en N-partes, y a cada una de ellas,
    # Como primer approach, partimos en N partes de igual tamaño, disjuntas
    # extenderlas según (6).
    parts_histo, parts_limits = split_extend_histo(histo_i, N)
    histo_target = [None]*N # array vacio para guardar los target histo
    w_k_functs = [None]*N # array vacio para guardar las funciones de peso
    histo_weights_values = [None]*N
    # previo_a_3 : TODO: Computar M_i y M_c según el paper, los cuales son
    # los parámetros alpha y beta
    M_i = dameM_i(histo_i)
    M_i = np.amax([M_i, 0.05])
    M_c = dameM_c(img_hsi[:,:,2],histo_i)
    M_c = np.amin([M_c, 1.])
    print('Mi:',M_i)
    print('Mc:',M_c)
    alpha = M_i/(M_i+M_c)
    beta = M_c/(M_i+M_c)
    # 3 : Aplicar HE a cada histrograma particionado según el paper:
    for i in range(0,N): # para cada partición del histograma

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
        # b : Resolver el sistema lineal, para hallar el target_histogram
        term_1 = np.multiply((alpha + beta), ident) + np.dot(np.multiply(gamma, D_T), D)
        term_1 = np.linalg.inv(term_1)
        # aca normalizo parts_histo[i]
        parts_histo[i] = np.divide(parts_histo[i],num_pixels)
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
    # TODO: esto de acá no debería tener que hacerse. debería dar ya una
    # distribucion
    total = sum(histo_equ)
    histo_equ /= total
    # 5 : Obtener el canal-I final, por HM
    # begin debug code --------------------------------
    # plt.subplot(1,2,1)
    # plt.imshow(np.divide(img_hsi[:,:,2],255), cmap='gray', vmin=0, vmax=1)

    img_hsi[:,:,2] = images.HM(img_hsi[:,:,2], histo_equ)

    # plt.subplot(1,2,2)
    # plt.imshow(img_hsi[:,:,2], cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # end debug code ----------------------------------

    # 6 : Convertir denuevo a RGB
    img_rgb_equ = converter.HSI2RGB(img_hsi)
    return img_rgb_equ


# Parte el histograma histo en N partes, y lo extiende según (6) en el paper
def split_extend_histo(histo, N):
    histo_parts = [None]*N
    histo_limits = [None]*N
    step = int(256/N)
    init = 0
    end = step
    for i in range(0,N):
        if i == N-1:
            end = 255 # el end de la ultima particion es directo 255
        # inicio una de las partes en 0's
        histo_parts[i] = np.zeros(256)
        for j in range(int(init), int(end)):
            # copio la parte que se encuentra dentro del rango
            histo_parts[i][j] = histo[j]
        histo_limits[i] = (init,end)
        init += step
        end += step
    # begin debug code --------------------------------
    # histo_check = np.zeros(256, dtype = np.uint)
    # for i in range(0,256):
    #     histo_check[i] = sum([histo_parts[j][i] for j in range(0,N)])
    #
    # print("original I-channel histo: \n")
    # print(histo)
    # print("equality histo: \n")
    # print(np.equal(histo_check, histo))
    # print(histo_parts)
    # end debug code ----------------------------------
    return histo_parts,histo_limits

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

def sobel_convolution(img):
    G_x = np.empty(img.shape)
    G_y = np.empty(img.shape)
    S = [[1,0,-1],[2,0,-2],[1,0,-1]]
    S_t = [[1,2,1],[0,0,0],[-1,-0,-1]]
    for x in range(1,len(img)-1):
        gx = 0
        gy = 0
        for y in range(1,len(img[0])-1):
            gx = -img[x-1][y-1]+img[x+1][y-1] + -2*img[x-1][y]+2*img[x+1][y] +-img[x-1][y+1]+img[x+1][y+1]
            G_x[x][y] = gx
            gy = -img[x-1][y-1]+img[x-1][y+1] + -2*img[x][y-1]+2*img[x][y+1] +-img[x+1][y-1]+img[x+1][y+1]
            G_y[x][y] = gy
    for x in range(len(img)):
        for y in range(len(img[0])):
            G_x[x][y] = np.sqrt(G_x[x][y]**2+G_x[x][y]**2)/(img[x][y]+1)

    return G_x
