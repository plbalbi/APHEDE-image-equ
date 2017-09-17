import aephe
from scipy import ndimage
from matplotlib import pyplot as plt
import sys


route = sys.argv[1]
print("Reading image...")
a = ndimage.imread(route)
plt.subplot(1,2,1)
plt.imshow(a)


print("Running AEPHE...")
b = aephe.AEPHE(a, 3)
plt.subplot(1,2,2)
plt.imshow(b)
plt.show()
