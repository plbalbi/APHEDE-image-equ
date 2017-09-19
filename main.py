import aephe
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Error, faltan argumentos.")
    print("%s imagen salida [N=3] " % (sys.argv[0]))
    # print("%s imagen salida [N=3] [alpha=.6] [beta=.2] [gamma=0]" % (sys.argv[0]))
    sys.exit(1)

route = sys.argv[1]
print("Reading image...")
img = ndimage.imread(route)

N = 3
a = .5
b = .5
g = 0.

if len(sys.argv) > 2:
    out_name = sys.argv[2]
    N = int(sys.argv[3])
    # N = int(sys.argv[2])
    # a = float(sys.argv[3])
    # b = float(sys.argv[4])
    # g = float(sys.argv[5])

print("Running AEPHE...")
img_AEPHE = aephe.AEPHE(img, N, alpha = a, beta = b, gamma = g)
print("Running AEPHE with acum_split...")
img_AEPHE_acum_splits = aephe.AEPHE(img, N, alpha = a, beta = b, gamma = g, acum_split=True)
plt.clf()
plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(img_AEPHE)
misc.imsave("out/%s_AEPHE.bmp" % (out_name), img_AEPHE)
plt.subplot(1,3,3)
plt.imshow(img_AEPHE_acum_splits)
misc.imsave("out/%s_AEPHE_AS.bmp" % (out_name), img_AEPHE_acum_splits)
plt.show()
