# AEPHE: Adaptative extended piecewise histogram equalisation

# Funcion principal
def AEPHE(img, N):
    # 1 : Transformar la imagen a HSI, computar el histograma del canal I.
    # 2 : Particionar el histograma recien computado en N-partes, y a cada una de ellas,
    # extenderlas según (6).
    # previo_a_3 : TODO: Computar M_i y M_c según el paper, los cuales son los parámetros alpha y beta
    # 3 : Aplicar HE a cada histrograma particionado según el paper:
        # a : Crear el histograma uniforme según la función de peso
        # b : Resolver el sistema lineal, para hallar el target_histogram
    # 4 : Juntar los histogramas una vez ecualizados, por el peso
    # 5 : Obtener el canal-I final, por HM
    # 6 : Convertir denuevo a RGB

