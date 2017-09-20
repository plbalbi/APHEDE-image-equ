import numpy as np
import math

def RGB2HSI(rgb):
    hsi = np.empty((len(rgb),len(rgb[0]),3),dtype=np.uint8)
    for f in range(len(rgb)):
        for j in range(len(rgb[f])):
            r = rgb[f][j][0]
            g = rgb[f][j][1]
            b = rgb[f][j][2]
            [h,s,i] = pix_RGB2HSI(r,g,b)
            hsi[f][j][0] = h
            hsi[f][j][1] = s
            hsi[f][j][2] = i
    return hsi


def HSI2RGB(hsi):
    rgb = np.empty(hsi.shape,dtype=np.uint8)
    for f in range(len(hsi)):
        for j in range(len(hsi[f])):
            h = hsi[f][j][0]
            s = hsi[f][j][1]
            i = hsi[f][j][2]
            [r,g,b] = pix_HSI2RGB(h,s,i)
            rgb[f][j][0] = r
            rgb[f][j][1] = g
            rgb[f][j][2] = b
    return rgb


def pix_RGB2HSI(R,G,B):
    r = R/255
    g = G/255
    b = B/255

    i = (r+g+b)/3
    s = 1-3*min(r,min(g,b))/(r+g+b+0.000001) # evitar dividir por 0
    h = 0
    denominador = ((r-g)**2+(r-b)*(g-b))**0.5 + 0.000001 # evitar dividir por 0
    numerador = 0.5*((r-g)+(r-b))
    tita = np.arccos(numerador/denominador)
    if b <= g:
        h = tita
    else:
        h = 2*math.pi - tita
    h = int(round(h*255/(2*math.pi)))
    s = int(round(s*255))
    i = int(round(i*255))
    return [h,s,i]


def pix_HSI2RGB(h,s,i):
    h = h*2*math.pi/255
    s = s/255
    i = i/255
    if h < 2*math.pi/3:
        r = min(1,i*(1+s*np.cos(h)/np.cos(math.pi/3-h)))
        b = i*(1-s)
        g = min(1,3*i-(r+b))
    else:
        if h < 4*math.pi/3:
            g = min(1,i*(1+s*np.cos(h - 2*math.pi/3)/np.cos(math.pi-h)))
            r = i*(1-s)
            b = min(1,3*i-(r+g))
        else:
            b = min(1,i*(1+s*np.cos(h-4*math.pi/3)/np.cos(5*math.pi/3-h)))
            g = i*(1-s)
            r = min(1,3*i-(g+b))
    r = int(round(r*255))
    g = int(round(g*255))
    b = int(round(b*255))
    return [r,g,b]

# Cosas que no se si van...
# Borrar?----------------------------------------------------------------------
def multiplicar_saturado(img,c):
    for f in range(len(img)):
        for j in range(len(img[f])):
            img[f][j] = np.uint8(np.clip(img[f][j]*c,0,255))
    return img

def sumar_saturado(img,c):
    for f in range(len(img)):
        for j in range(len(img[f])):
            img[f][j] = min(255,img[f][j]+c)
    return img

def sumar_circular(img,c):
    for f in range(len(img)):
        for j in range(len(img[f])):
            img[f][j] = (img[f][j]+c)%256
    return img

def multiplicar_circular(img,c):
    for f in range(len(img)):
        for j in range(len(img[f])):
            val = img[f][j]*c
            while val > 255:
                val -= 255
            img[f][j] = val
    return img
# Borrar?----------------------------------------------------------------------
