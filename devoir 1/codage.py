import numpy as np

def normalisation(data):
    
    data_meaned = data - data.mean(0) # si pas d'argument ça fait un scalaire puis 0 ou 1 c'est le numéro de la dimension sur la quelle la moyenne est faite en gros.
    data_std = data_meaned / data_meaned.std(0)
    return data_std
    
def quantitatif_en_qualitatif1(data, ninter):
    mat_interv = (data.max(0) - data.min(0)) / ninter
    bornes = [data.min(0) + i * mat_interv for i in range(ninter)]
    bornes.append(data.max(0))

    mat_qual1 = np.zeros(data.shape, dtype='int')
    for i in range(data.shape[0]):
        for k in range(data.shape[1]):
            j = 0
            trouve = False
            while not trouve and j < ninter:
                if data[i, k] >= bornes[j][k] and data[i, k] <= bornes[j + 1][k]:
                    trouve = True
                else:
                    j += 1
            mat_qual1[i, k] = j

    return mat_qual1

def quantitatif_en_qualitatif2(data, nbclasses):

    I = np.argsort(data,axis=0)

    nb_par_classe = data.shape[0] // nbclasses
    
    inddeb = 0
    classes = np.zeros(data.shape[0],dtype='int')
    for i in range(nbclasses):
        indfin = inddeb+nb_par_classe

        classes[inddeb:indfin] = i
        inddeb = indfin

    matqual2 = np.zeros(data.shape,dtype='int')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            matqual2[I[i,j],j] = classes[i]
            
    return matqual2


    