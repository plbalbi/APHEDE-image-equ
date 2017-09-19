import aephe
from scipy import ndimage
from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Error, faltan argumentos.")
    print("%s imagen [N=3] [alpha=.6] [beta=.2] [gamma=0]" % (sys.argv[0]))
    sys.exit(1)

route = sys.argv[1]
print("Reading image...")
img = ndimage.imread(route)

N = 3
a = .5
b = .5
g = 0.

if len(sys.argv) > 2:
    N = int(sys.argv[2])
    a = float(sys.argv[3])
    b = float(sys.argv[4])
    g = float(sys.argv[5])

print("Running AEPHE...")
img_AEPHE = aephe.AEPHE(img, N, alpha = a, beta = b, gamma = g, splits = [100,200])
plt.clf()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_AEPHE)
plt.show()
