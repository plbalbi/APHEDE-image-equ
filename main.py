import aephe
from scipy import ndimage
from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Error, faltan argumentos. Úsese alguna de las variantes:")
    print("%s imagen [N=3] [alpha=.6] [beta=.2] [gamma=0]" % (sys.argv[0]))
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
if len(sys.argv) > 2:
    if sys.argv[2]=='-s':
        splits = []
        for i in range(3,len(sys.argv)):
            splits.append(int(sys.argv[i]))
        print(splits)
    else:
        N = int(sys.argv[2])
        a = float(sys.argv[3])
        b = float(sys.argv[4])
        g = float(sys.argv[5])

print("Running AEPHE...")
img_AEPHE = aephe.AEPHE(img, N, alpha = a, beta = b, gamma = g, splits = splits,plot=True)
