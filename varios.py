import matplotlib
from matplotlib import pyplot as plt
import images
import converter
from scipy import misc
import seaborn as sns
sns.set_style("dark")


def print_acum(img):
    histo = images.get_histo(img)
    for i in range(1,256):
        histo[i] += histo[i-1]
    plt.plot(range(0,256), histo)
    plt.xlim(0,255)
    plt.tick_params(labelsize = 10)
    plt.savefig("informe/imgs/acum_sam.pdf")


img = misc.imread('../fotos/sam_1_chico.JPG')
img_hsi = converter.RGB2HSI(img)

print_acum(img_hsi[:,:,2])


