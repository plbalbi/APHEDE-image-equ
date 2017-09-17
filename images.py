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

