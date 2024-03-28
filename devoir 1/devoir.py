import codage as cd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans2

def readfile(filename):
	with open(filename) as f:
		data = f.read().splitlines()

	return data

def acp(X,nom_individus,nom_variables):
    
    # Calcul de la matrice M (métrique)
    M = np.eye(X.shape[1])
    
    # Calcul de la matrice D (poid des individus)
    D = np.eye(X.shape[0]) / X.shape[0]
    
    # Calcul de la matrice de covariance pour les individus
    Xcov_ind = X.T.dot(D.dot(X.dot(M))) #X'DXM
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    val_p_ind = np.sort(L)[::-1] #valeurs de u
    vect_p_ind = U[:,indices] #u
    
    # Calcul des facteurs pour les individus  Fu = XMu 
    fact_ind = X.dot(M.dot(vect_p_ind))
    
    # Calcul des facteurs pour les variables actives (utilisation des relations de transition) 
    fact_var = X.T.dot(D.dot(fact_ind)) / np.sqrt(val_p_ind) #dualité, on divise par lambda_s

    # Calcul de la contribution des individus aux axes factoriels
    contributions_ind = np.zeros(fact_ind.shape) 
    for i in range(fact_ind.shape[1]):  
        f = fact_ind[:,i]
        contributions_ind[:,i] = 100 * D.dot(f*f) / f.T.dot(D.dot(f)) #Inertie du nuage projeté (Fu'DFu)
    
    # Calcul des pourcentage d'inertie des axes factoriels
    inerties = 100*val_p_ind / val_p_ind.sum()
    
    print('Pourcentages d"inertie :')
    print(inerties)
    
    # Affichage du diagramme d'inertie
    plt.figure(1)
    plt.plot(inerties,'o-')
    plt.title('Diagramme des inerties')
    
    # Affichage du plan factoriel
    plt.figure(2)
    x = np.arange(-1,1,0.001)
    cercle_unite = np.zeros((2,len(x)))
    cercle_unite[0,:] = np.sqrt(1-x**2)
    cercle_unite[1,:] = -cercle_unite[0,:]
    plt.plot(x,cercle_unite[0,:])
    plt.plot(x,cercle_unite[1,:])
    plt.plot(fact_var[:,0],fact_var[:,1],'x')
    plt.yscale('linear')
    plt.ylim(-1.2,1.2)
    plt.xlim(-1.2,1.2)
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACP Projection des variables')
    
    for label,x,y in zip(nom_variables,fact_var[:,0],fact_var[:,1]):
        plt.annotate(label,
                     xy = (x,y),
                     xytext = (-50,5),
                     textcoords = 'offset points',
                     arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0')
                     )

    # Affichage du plan factoriel pour les individus
    plt.figure(3)
    plt.plot(fact_ind[:,0],fact_ind[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACP Projection des individus')
    for label,x,y in zip(nom_individus,fact_ind[:,0],fact_ind[:,1]):
        plt.annotate(label,
                     color = 'r',
                     xy=(x,y),
                     xytext=(-10,2),
                     textcoords="offset points"
                     )
    
    plt.show()

def acp_cah(X, nom_individus, nom_variables) : 
    
    # Calcul de la matrice M (métrique)
    M = np.eye(X.shape[1])
    
    # Calcul de la matrice D (poid des individus)
    D = np.eye(X.shape[0]) / X.shape[0]
    
    # Calcul de la matrice de covariance pour les individus
    Xcov_ind = X.T.dot(D.dot(X.dot(M))) #X'DXM
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    vect_p_ind = U[:,indices] #u
    
    # Calcul des facteurs pour les individus  Fu = XMu 
    fact_ind = X.dot(M.dot(vect_p_ind))

    X = fact_ind[:,:2] #On réalise la CAH sur les 2 premiers facteurs pour avoir une analyse claire
    
    Z = linkage(X, 'ward')

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
	    Z,
	    leaf_rotation=90.,  
	    leaf_font_size=8.,
	)

    k=4
    clusters = fcluster(Z, k, criterion='distance')
    print(clusters)
    
    
    c1, c2, c3, c4, c5, c6 = [],[],[],[],[],[]
    for i in range(len(nom_individus)) :  #Je crée juste des listes avec les éléments des clusters pour pouvoir les faire apparaître sur le graph de l'ACP
        if clusters[i] == 1 :
            c1.append(nom_individus[i])
        elif clusters[i] == 2 :
            c2.append(nom_individus[i])
        elif clusters[i] == 3 : 
            c3.append(nom_individus[i])
        elif clusters[i] == 4 : 
            c4.append(nom_individus[i])
        elif clusters[i] == 5 : 
            c5.append(nom_individus[i])
        elif clusters[i] == 6 : 
            c6.append(nom_individus[i])
    print(c1, c2, c3, c4 , c5, c6)
            

    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:,1], c=clusters)

    for label,x,y in zip(nom_individus,X[:,0],X[:,1]):
        plt.annotate(label,
			xy=(x,y),
			xytext=(-50,5),
			textcoords='offset points',
			ha='right', va='bottom',
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
			)
		
    plt.title('Classification ascendante hierarchique')
    plt.show()
    
    # Affichage du plan factoriel pour les individus après avoir fait la CAH
    plt.figure(3)
    plt.plot(fact_ind[:,0],fact_ind[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACP Projection des individus après la CAH')
    for label,x,y in zip(nom_individus,fact_ind[:,0],fact_ind[:,1]):
        if label in c1 :
            couleur = 'r'
        if label in c2 :
            couleur = 'g'
        if label in c3 : 
            couleur = 'b'
        if label in c4 : 
            couleur = 'black'
        if label in c5 : 
            couleur = 'purple'
        if label in c6 : 
            couleur = 'orange'
        plt.annotate(label, xy=(x,y), color = couleur, xytext=(-10,2), textcoords="offset points")
    
    plt.show()
    
def acp_kmeans(X, nom_individus, nom_variables) : 
    
    # Calcul de la matrice M (métrique)
    M = np.eye(X.shape[1])
    
    # Calcul de la matrice D (poid des individus)
    D = np.eye(X.shape[0]) / X.shape[0]
    
    # Calcul de la matrice de covariance pour les individus
    Xcov_ind = X.T.dot(D.dot(X.dot(M))) #X'DXM
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    vect_p_ind = U[:,indices] #u
    
    # Calcul des facteurs pour les individus  Fu = XMu 
    fact_ind = X.dot(M.dot(vect_p_ind))
    
    X = fact_ind[:,:2] #Idem on réalise sur les deux premiers facteurs pour comparer les résultats
    #k = 3 #On fixe k le nombre de clusters à réaliser
    c1, c2, c3 = [],[],[] #Initialisation des clusters
    centre_init = np.array([X[23],
                           X[22],
                           X[10]])
    print(centre_init)
    centroids, indice = kmeans2(X,centre_init, iter=100, minit='matrix')
    for i in range(len(indice)) :
        if indice[i] == 0 : 
            c1.append(nom_individus[i])
        elif indice[i] == 1 :
            c2.append(nom_individus[i])
        else :
            c3.append(nom_individus[i])
            
    print (c1, c2, c3)
    
    plt.figure() #On représente l'ACP avec les clusters obtenus pour les deux premiers facteurs 
    plt.plot(fact_ind[:,0],fact_ind[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACP Projection des individus après les kmeans')
    for label,x,y in zip(nom_individus,fact_ind[:,0],fact_ind[:,1]):
        if label in c1 :
            couleur = 'r'
        if label in c2 :
            couleur = 'g'
        if label in c3 : 
            couleur = 'b'
        plt.annotate(label, xy=(x,y), color = couleur, xytext=(-10,2), textcoords="offset points")
    plt.show()
    
def acm(X, nom_individus, nom_variables) :
    
    nb_modalites_par_var = X.max(0) 
    nb_modalites = int(nb_modalites_par_var.sum())
    
    XTDC = np.zeros((X.shape[0],nb_modalites))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            XTDC[i, int(nb_modalites_par_var[:j].sum() + X[i,j]-1)] = 1
            
    nom_modalites = []
    for i in range(data.shape[1]):
        for j in range(int(nb_modalites_par_var[i])):
            nom_modalites.append(nom_variables[i]+str(j+1))
    
    Xfreq = XTDC / XTDC.sum()
    
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0],1)
    marge_ligne = Xfreq.sum(0).reshape(1,Xfreq.shape[1])
    Xindep = marge_ligne * marge_colonne   
    
    X = Xfreq / Xindep - 1
    
    M = np.diag(marge_ligne[0,:])
    D = np.diag(marge_colonne[:,0])
    
    # Calcul de la matrice de covariance pour les modalités en ligne
    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    print(type(indices))
    val_p_ind = np.float16(np.sort(L)[::-1]) # car des valeurs peuvent être complexes avec une partie imaginaire nulle en raison d'approximations numériques
    vect_p_ind = U[:,indices]
    
    # Suppression des éventuelles valeurs propres nulles
    indices = np.nonzero(val_p_ind > 0)[0]
    print(type(indices))
    val_p_ind = val_p_ind[indices]
    vect_p_ind = vect_p_ind[:,indices]
    
    # Calcul des facteurs pour les modalités en ligne 
    fact_ind = X.dot(M.dot(vect_p_ind))
    
    # Calcul des facteurs pour les modalités en colonne
    fact_mod = X.T.dot(D.dot(fact_ind)) / np.sqrt(val_p_ind)
    
    # Calcul des pourcentage d'inertie des axes factoriels
    inerties = 100*val_p_ind / val_p_ind.sum()
    
    print('Pourcentages d"inertie :')
    print(inerties)
    
    # Affichage du diagramme d'inertie
    plt.figure(1)
    plt.plot(inerties,'o-')
    plt.title('Diagramme des inerties')
    
    # Affichage du plan factoriel pour les indidivus
    plt.figure(2)
    plt.plot(fact_ind[:,0],fact_ind[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACM Projection des individus')
    for label,x,y in zip(nom_individus,fact_ind[:,0],fact_ind[:,1]):
        plt.annotate(label,
                      xy=(x,y),
                      xytext=(-10,2),
                      textcoords="offset points"
                      )
    
    # Affichage du plan factoriel pour les modalités en colonne
    plt.figure(3)
    plt.plot(fact_mod[:,0],fact_mod[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACM Projection des modalités')
    for label,x,y in zip(nom_modalites,fact_mod[:,0],fact_mod[:,1]):
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(-10,2),
                     textcoords="offset points"
                     )
    plt.show()
    
def acm_cah(X, nom_individus, nom_variables):
    
    nb_modalites_par_var = X.max(0) 
    nb_modalites = int(nb_modalites_par_var.sum())
    
    XTDC = np.zeros((X.shape[0],nb_modalites))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            XTDC[i, int(nb_modalites_par_var[:j].sum() + X[i,j]-1)] = 1
            
    nom_modalites = []
    for i in range(X.shape[1]):
        for j in range(int(nb_modalites_par_var[i])):
            nom_modalites.append(nom_variables[i]+str(j+1))
    
    Xfreq = XTDC / XTDC.sum()
    
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0],1)
    marge_ligne = Xfreq.sum(0).reshape(1,Xfreq.shape[1])
    Xindep = marge_ligne * marge_colonne   
    
    X = Xfreq / Xindep - 1
    
    M = np.diag(marge_ligne[0,:])
    D = np.diag(marge_colonne[:,0])
    
    # Calcul de la matrice de covariance pour les modalités en ligne
    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    print(type(indices))
    val_p_ind = np.float16(np.sort(L)[::-1]) # car des valeurs peuvent être complexes avec une partie imaginaire nulle en raison d'approximations numériques
    vect_p_ind = U[:,indices]
    
    # Suppression des éventuelles valeurs propres nulles
    indices = np.nonzero(val_p_ind > 0)[0]
    print(type(indices))
    val_p_ind = val_p_ind[indices]
    vect_p_ind = vect_p_ind[:,indices]
    
    # Calcul des facteurs pour les modalités en ligne 
    fact_ind = X.dot(M.dot(vect_p_ind))
    
    # Calcul des facteurs pour les modalités en colonne
    fact_mod = X.T.dot(D.dot(fact_ind)) / np.sqrt(val_p_ind)
    
    Xind = fact_ind[:,:2] #On réalise la CAH sur les 2 premiers facteurs pour avoir une analyse filtrée
    Xmod = fact_mod[:,:2]
    
    Zind = linkage(Xind, 'ward')
    Zmod = linkage(Xmod, 'ward')

    plt.title('Dendogram individus')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
	    Zind,
	    leaf_rotation=90.,  
	    leaf_font_size=8.,
	)
    plt.show()

    plt.title('Dendrogram modalités')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
	    Zmod,
	    leaf_rotation=90.,  
	    leaf_font_size=8.,
	)
    plt.show()
    
    k=3
    clusters_ind = fcluster(Zind, k, criterion='maxclust')
    print(clusters_ind)
    
    
    c1_ind, c2_ind, c3_ind=[],[],[]
    for i in range(len(nom_individus)) :  #Je crée juste des listes avec les éléments des clusters pour pouvoir les faire apparaître sur le graph de l'ACP
        if clusters_ind[i] == 1 :
            c1_ind.append(nom_individus[i])
        elif clusters_ind[i] == 2 :
            c2_ind.append(nom_individus[i])
        elif clusters_ind[i] == 3 : 
            c3_ind.append(nom_individus[i])
    print(c1_ind, c2_ind, c3_ind)
    
    clusters_mod = fcluster(Zmod, k, criterion='maxclust')
    print(clusters_mod)
    
    c1_mod, c2_mod, c3_mod=[],[],[]
    for i in range(len(nom_modalites)) :  #Je crée juste des listes avec les éléments des clusters pour pouvoir les faire apparaître sur le graph de l'ACP
        if clusters_mod[i] == 1 :
            c1_mod.append(nom_modalites[i])
        elif clusters_mod[i] == 2 :
            c2_mod.append(nom_modalites[i])
        elif clusters_mod[i] == 3 : 
            c3_mod.append(nom_modalites[i])
    print(c1_mod, c2_mod, c3_mod)
            

    plt.figure(figsize=(10, 8))
    plt.scatter(Xind[:,0], Xind[:,1], c=clusters_ind)

    for label,x,y in zip(nom_individus, Xind[:,0], Xind[:,1]):
        plt.annotate(label,
			xy=(x,y),
			xytext=(-50,5),
			textcoords='offset points',
			ha='right', va='bottom',
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
			)
		
    plt.title('Classification ascendante hierarchique pour les individus')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(Xmod[:,0], Xmod[:,1], c=clusters_mod)

    for label,x,y in zip(nom_modalites, Xmod[:,0], Xmod[:,1]):
        plt.annotate(label,
			xy=(x,y),
			xytext=(-50,5),
			textcoords='offset points',
			ha='right', va='bottom',
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
			)
		
    plt.title('Classification ascendante hierarchique pour les modalités')
    plt.show()
    
    
    
    # Affichage du plan factoriel pour les individus après avoir fait la CAH
    plt.figure(3)
    plt.plot(fact_ind[:,0],fact_ind[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACM Projection des individus après la CAH')
    for label,x,y in zip(nom_individus,fact_ind[:,0],fact_ind[:,1]):
        if label in c1_ind :
            couleur = 'r'
        if label in c2_ind :
            couleur = 'g'
        if label in c3_ind : 
            couleur = 'b'
        plt.annotate(label, xy=(x,y), color = couleur, xytext=(-10,2), textcoords="offset points")
    
    plt.show()
    
    # Affichage du plan factoriel pour les individus après avoir fait la CAH
    plt.figure(4)
    plt.plot(fact_mod[:,0],fact_mod[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACM Projection des modalités après la CAH')
    for label,x,y in zip(nom_modalites,fact_mod[:,0],fact_mod[:,1]):
        if label in c1_mod :
            couleur = 'orange'
        if label in c2_mod :
            couleur = 'purple'
        if label in c3_mod : 
            couleur = 'cyan'
        plt.annotate(label, xy=(x,y), color = couleur, xytext=(-10,2), textcoords="offset points")
    
    plt.show()

def acm_kmeans(X, nom_individus, nom_variables) :
    
    nb_modalites_par_var = X.max(0) 
    nb_modalites = int(nb_modalites_par_var.sum())
    
    XTDC = np.zeros((X.shape[0],nb_modalites))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            XTDC[i, int(nb_modalites_par_var[:j].sum() + X[i,j]-1)] = 1
            
    nom_modalites = []
    for i in range(X.shape[1]):
        for j in range(int(nb_modalites_par_var[i])):
            nom_modalites.append(nom_variables[i]+str(j+1))
    
    Xfreq = XTDC / XTDC.sum()
    
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0],1)
    marge_ligne = Xfreq.sum(0).reshape(1,Xfreq.shape[1])
    Xindep = marge_ligne * marge_colonne   
    
    X = Xfreq / Xindep - 1
    
    M = np.diag(marge_ligne[0,:])
    D = np.diag(marge_colonne[:,0])
    
    # Calcul de la matrice de covariance pour les modalités en ligne
    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    print(type(indices))
    val_p_ind = np.float16(np.sort(L)[::-1]) # car des valeurs peuvent être complexes avec une partie imaginaire nulle en raison d'approximations numériques
    vect_p_ind = U[:,indices]
    
    # Suppression des éventuelles valeurs propres nulles
    indices = np.nonzero(val_p_ind > 0)[0]
    print(type(indices))
    val_p_ind = val_p_ind[indices]
    vect_p_ind = vect_p_ind[:,indices]
    
    # Calcul des facteurs pour les modalités en ligne 
    fact_ind = X.dot(M.dot(vect_p_ind))
    
    # Calcul des facteurs pour les modalités en colonne
    fact_mod = X.T.dot(D.dot(fact_ind)) / np.sqrt(val_p_ind)
    
    Xind = fact_ind[:,:2] #On réalise les kmeans sur les 2 premiers facteurs pour avoir une analyse filtrée
    Xmod = fact_mod[:,:2]
    
    Xind = Xind.astype(float) #J'ai du rajouter cela car j'avais des +0j dans les valeurs de Xind, je ne comprends pas pourquoi ces +0j étaient apparus 
    Xmod = Xmod.astype(float) #malgré le np.float16 ajouté sur le vecteur des valeurs propres 
    
    
    centre_init_ind = (np.array([Xind[-1],
                                Xind[9],
                                Xind[-4]]))
    
    centre_init_mod = np.array([Xmod[9],
                                Xmod[10],
                                Xmod[6]])
    
    print(centre_init_ind, centre_init_mod)
    
    centroids_ind, indice_ind = kmeans2(Xind, centre_init_ind, iter=100, minit='matrix')
    centroids_mod, indice_mod = kmeans2(Xmod, centre_init_mod, iter=100, minit='matrix')
    
    c1_ind, c2_ind, c3_ind, c1_mod, c2_mod, c3_mod = [],[],[],[],[],[]
    
    for i in range(len(indice_ind)) : 
        if indice_ind[i] == 0 :
            c1_ind.append(nom_individus[i])
        if indice_ind[i]== 1: 
            c2_ind.append(nom_individus[i])
        if indice_ind[i]==2 :
            c3_ind.append(nom_individus[i])
    
    for i in range(len(indice_mod)) :
        if indice_mod[i] == 0 :
            c1_mod.append(nom_modalites[i])
        if indice_mod[i] == 1 : 
            c2_mod.append(nom_modalites[i])
        if indice_mod[i] == 2 : 
            c3_mod.append(nom_modalites[i])
            
    print(c1_ind, c2_ind, c3_ind)
    print(c1_mod, c2_mod, c3_mod)
    # Affichage du plan factoriel pour les individus après avoir fait la CAH
    plt.figure(3)
    plt.plot(fact_ind[:,0],fact_ind[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACM Projection des individus après la CAH')
    for label,x,y in zip(nom_individus,fact_ind[:,0],fact_ind[:,1]):
        if label in c1_ind :
            couleur = 'r'
        if label in c2_ind :
            couleur = 'g'
        if label in c3_ind : 
            couleur = 'b'
        plt.annotate(label, xy=(x,y), color = couleur, xytext=(-10,2), textcoords="offset points")
    
    plt.show()
    
    # Affichage du plan factoriel pour les individus après avoir fait la CAH
    plt.figure(4)
    plt.plot(fact_mod[:,0],fact_mod[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACM Projection des modalités après la CAH')
    for label,x,y in zip(nom_modalites,fact_mod[:,0],fact_mod[:,1]):
        if label in c1_mod :
            couleur = 'orange'
        if label in c2_mod :
            couleur = 'purple'
        if label in c3_mod : 
            couleur = 'cyan'
        plt.annotate(label, xy=(x,y), color = couleur, xytext=(-10,2), textcoords="offset points")
    
    plt.show()
    
        
        


if __name__ == '__main__' :
    
    # Lecture des données à partir des fichiers textes
    data = np.loadtxt('donnees/population_donnees.txt')
    nom_individus = readfile('donnees/population_noms_individus.txt')
    nom_variables = readfile('donnees/population_noms_variables.txt')
    
    # Normalisation des données
    data = cd.normalisation(data)
    data_acm = cd.quantitatif_en_qualitatif1(data, 4)+1 #Je rends la matrice compatible avec l'ACM et je fais +1 pour pas qu'il croit qu'il y a que 3 valeurs 
    
    #Pas de variables illustratives donc pas besoin de séparer en X et Xsup comme cela avait été fait au TD2

    # Réalisation de l'ACP
    #acp(data,nom_individus,nom_variables)
    
    # Réalisation de l'ACP CAH
    #acp_cah(data, nom_individus, nom_variables)

