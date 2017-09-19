import aephe
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Error, faltan argumentos. Úsese alguna de las variantes:")
    print("%s imagen out N" % (sys.argv[0]))
    # print("%s imagen [N=3] [alpha=.6] [beta=.2] [gamma=0]" % (sys.argv[0]))
    print("%s imagen -s splits= 85 170" % (sys.argv[0]))
    sys.exit(1)

route = sys.argv[1]
print("Reading image...")
img = ndimage.imread(route)

N = 3
a = .5
b = .5
g = 0.
splits=None
out_name = None
if len(sys.argv) > 2:
    if sys.argv[2]=='-s':
        splits = []
        for i in range(3,len(sys.argv)):
            splits.append(int(sys.argv[i]))
        print(splits)
    else:
        N = int(sys.argv[3])
        out_name = sys.argv[2]
        # N = int(sys.argv[2])
        # a = float(sys.argv[3])
        # b = float(sys.argv[4])
        # g = float(sys.argv[5])

print("Running AEPHE...")
img_AEPHE = aephe.AEPHE(img, N, alpha = a, beta = b, gamma = g, splits = splits, plot=True)
print("Running AEPHE with acum_split...")
img_AEPHE_acum_splits = aephe.AEPHE(img, N, alpha = a, beta = b, gamma = g, splits = splits, plot=True, acum_split=True)
plt.clf()
plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(img_AEPHE)
plt.subplot(1,3,3)
plt.imshow(img_AEPHE_acum_splits)
plt.show()
if out_name != None:
    misc.imsave("out/%s_AEPHE.bmp" % (out_name), img_AEPHE)
    misc.imsave("out/%s_AEPHE_AS.bmp" % (out_name), img_AEPHE_acum_splits)
