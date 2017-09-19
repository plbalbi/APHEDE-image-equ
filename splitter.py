import numpy as np
from matplotlib import pyplot as plt

# Parte el histograma histo en N partes iguales, y lo extiende según (6)
def split_extend_histo(histo, N):
    histo_parts = [None]*N
    histo_limits = [None]*N
    step = int(256/N)
    init = 0
    end = step
    for i in range(0,N):
        if i == N-1:
            end = 256 # el end de la ultima particion es directo 255
        # inicio una de las partes en 0's
        histo_parts[i] = np.zeros(256)
        for j in range(int(init), int(end)):
            # copio la parte que se encuentra dentro del rango
            histo_parts[i][j] = histo[j]
        histo_limits[i] = (init,end)
        init += step
        end += step
    # begin debug code --------------------------------
    # histo_check = np.zeros(256, dtype = np.uint)
    # for i in range(0,256):
    #     histo_check[i] = sum([histo_parts[j][i] for j in range(0,N)])
    #
    # print("original I-channel histo: \n")
    # print(histo)
    # print("equality histo: \n")
    # print(np.equal(histo_check, histo))
    # print(histo_parts)
    # end debug code ----------------------------------
    return histo_parts,histo_limits

# Parte el histograma histo en los lugares pasados
def custom_split_extend_histo(histo, splits,show=False):
    # verifico que los splits sean válidos
    for i in range(len(splits)-1):
        assert splits[i]<splits[i+1]
    for split in splits:
        assert split > 0 and split < 255

    N = len(splits)+1
    histo_parts = [[] for l in range(N)]
    histo_limits = [[] for l in range(N)]
    splits.append(256)
    init = 0
    for i in range(len(splits)):
        end = splits[i]
        # inicio una de las partes en 0's
        histo_parts[i] = np.zeros(256)
        for j in range(init,end):
            # copio la parte que se encuentra dentro del rango
            histo_parts[i][j] = histo[j]
        histo_limits[i] = (init,end)
        init = end
    # si me lo indican, muestro la partición del histograma
    if show:
        for i in range(len(histo_limits)-1):
            plt.axvline(x=histo_limits[i][1],color='r')
        plt.plot(histo)
        plt.show()
    return histo_parts, histo_limits
