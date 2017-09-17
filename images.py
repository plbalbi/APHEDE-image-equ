import numpy as np

L = 256

def get_histo(img):
    # Se asume que en la imágen que se pasa, sus valores están en el rango
    # [0, 255]
    h = np.zeros(L)
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            val = img[i][j]
            h[val] = h[val] + 1
    return h

def HM(img, histo):
    assert len(histo)==L

    # guardo las frecuencias absolutas de la imagen para calcular la acumulada
    acum_r = np.zeros(L)
    for f in range(len(img)):
        for j in range(len(img[f])):
            acum_r[img[f][j]]+=1

    # calculo las dos acumuladas, la entrada r, la normal s
    acum_r[0] /= float(len(img)*len(img[0]))
    for i in range(1,L):
        # calculo las frecuencias relativas, obteniendo la distribucion
        acum_r[i] /= float(len(img)*len(img[0]))
        # acumulo
        acum_r[i] += acum_r[i-1]
        histo[i] += histo[i-1]

    # calculo las imagenes de los valores 0...255 en la funcion T(r)
    # la salida s se calcula como s=T(r)=F_s^-1(F_r(r)).
    # al trabajar con valores discretos, se busca el valor mayor inmediato.
    T_r = np.zeros(L)
    for p in range(L):
        acum = acum_r[p]
        k = 0
        while histo[k]<acum and k<255:
            k+=1
        T_r[p] = k

    # construyo la imagen de salida
    for f in range(len(img)):
        for j in range(len(img[f])):
            img[f][j] = T_r[img[f][j]]

    return img
