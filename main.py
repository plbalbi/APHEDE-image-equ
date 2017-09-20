import aephe
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 3:
    print("Error, faltan argumentos. Uso:")
    print("%s imagen-in imagen-out" % (sys.argv[0]))
    sys.exit(1)

route = sys.argv[1]
print("Reading image...")
img = ndimage.imread(route)

N = 3
a = .3
b = .6
g = 0.
splits=None
out_name = sys.argv[2]

print("Running AEPHE...")
img_AEPHE_acum_splits = aephe.AEPHE(img, N, alpha = a, beta = b, gamma = g, splits = splits, acum_split=True)

# muestro comparación con imagen original
plt.clf()
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_AEPHE_acum_splits)
plt.title('Procesada AEPHE')
plt.show()

# guardo la salida en out/
if out_name != None:
    misc.imsave("out/%s_AEPHE.bmp" % (out_name), img_AEPHE_acum_splits)
