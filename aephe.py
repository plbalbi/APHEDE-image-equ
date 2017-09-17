import sys
import numpy as np
import converter
import images
# AEPHE: Adaptative extended piecewise histogram equalisation

# Funcion principal
# por defecto les paso un tercio, para que no explote si me olvido de pasarle algo
def AEPHE(img, N, alpha=1./3., beta=1./3., gamma=1./3.):
    # 1 : Transformar la imagen a HSI, computar el histograma del canal I.
    img_hsi = converter.RGB2HSI(img)
    histo_i = images.get_histo(img_hsi[:,:,2])
    num_pixels = len(img)*len(img[0]) # cantidad de pixels
    # 2 : Particionar el histograma recien computado en N-partes, y a cada una de ellas,
    # Como primer approach, partimos en N partes de igual tamaño, disjuntas
    # extenderlas según (6).
    parts_histo = split_extend_histo(histo_i, N)
    histo_target = [None]*N # array vacio para guardar los target histo
    w_k_functs = [None]*N # array vacio para guardar las funciones de peso
    histo_weights_values = [None]*N
        # previo_a_3 : TODO: Computar M_i y M_c según el paper, los cuales son
        # los parámetros alpha y beta
    # 3 : Aplicar HE a cada histrograma particionado según el paper:
    for i in range(0,N): # para cada partición del histograma

        # IMPORTANTE:
        # parts_histo[i] esta en float, pero sus valores son en int, hay que
        # normalizarlo antes de calcular el target_histo

        # a : Crear el histograma uniforme según la función de peso
        # w_k sería la funcion peso para generar el histograma uniforme
        w_k = weight_function(parts_histo[i])
        w_k_functs[i] = w_k # guardo la funcion de pesaje
        histo_unif = np.zeros(256) # esta en float
        curr_weights = np.zeros(256)
        for j in range(0,256):
            if parts_histo[i][j]>0.1: # si no es cero, quiere decir que el valor esta en
            # esta parte del histo
                histo_unif[j] = 1
            else:
                # el valor debe ser 0, es decir, no esta en la parte,
                # aplico weight_function
                histo_unif[j] = w_k(j)
            # calculo un array que tiene para cada nivel i, w_k_i
            # (weight_function of the piece-histogram) -> SE USA EN (4)
            curr_weights[j] = w_k(j)
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
        local_sum = 0
        for n in range(0,N):
            local_sum = local_sum + histo_weights_values[n][i]
        total_weights[i] = local_sum

    # merge de histogramas
    histo_equ = np.zeros(256)
    for i in range(0,256):
        # acumulo los histos con peso
        histo_equ[i] = sum([ histo_weights_values[j][i] / total_weights[i] * histo_target[j][i]\
                for j in range(0,N)])
    # 5 : Obtener el canal-I final, por HM
    img_hsi[:,:,2] = images.HM(img_hsi[:,:,2], histo_equ)

    # 6 : Convertir denuevo a RGB
    img_rgb_equ = converter.HSI2RGB(img_hsi)
    return img_rgb_equ


# Parte el histograma histo en N partes, y lo extiende según (6) en el paper
def split_extend_histo(histo, N):
    histo_parts = [None]*N
    step = np.floor(256./float(N))
    init = 0
    end = step
    for i in range(0,N):
        if i == N-1:
            end = 256 # el end de la ultima particion es directo 255
        # inicio una de las partes en 0's
        histo_parts[i] = np.zeros(256)
        for j in range(int(init), int(end)):
            # copio la parte que se encuentra dentro del rango
            histo_parts[i][j] = histo[j]
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
    count = 0.
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

    if count < .1:
        print("Count es cero!")
        print("Histo: ")
        print(piece_histo)
        print("\n\nFirst val: %d" % (first_val))
        print("\nEnd val: %d" % (last_val))
        sys.exit(0)

    mean = mean/count
    # sugerido por paper
    sigma_th = np.abs(128-mean)
    # cantidad de valores que cubre el intervalo
    w_d = last_val - first_val
    # tomo el maximo
    sigma_k = np.amax([w_d,sigma_th])
    return lambda i: np.exp(-((i - u_k)**2)/(2*(sigma_k**2)))
