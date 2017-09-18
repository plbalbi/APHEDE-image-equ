import aephe
from scipy import ndimage
from matplotlib import pyplot as plt
import sys


route = sys.argv[1]
print("Reading image...")
a = ndimage.imread(route)


print("Running AEPHE...")
b = aephe.AEPHE(a, 3 , alpha = .8, beta = .2, gamma = 0)
plt.clf()
plt.subplot(1,2,1)
plt.imshow(a)
plt.subplot(1,2,2)
plt.imshow(b)
plt.show()
