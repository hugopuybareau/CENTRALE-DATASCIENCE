import matplotlib.pyplot as plt
import numpy as np

def lecture_donnees(nom_fichier, delimiteur=','):
    """ Lit le fichier contenant les données et renvoiee les matrices correspondant

    Parametres
    ----------
    nom_fichier : nom du fichier contenant les données
    delimiteur : caratère délimitant les colonne dans le fichier ("," par défaut)

    Retour
    -------
    x : matrice des données de dimension [N, nb_var]
    d : matrice contenant les valeurs de la variable cible de dimension [N, nb_cible]
    N : nombre d'éléments
    nb_var : nombre de variables prédictives
    nb_cible : nombre de variables cibles

    """
    
    data = np.loadtxt(nom_fichier, delimiter=delimiteur)
    
    nb_cible = 1
    nb_var = data.shape[1] - nb_cible
    N = data.shape[0]

    x = data[:, :-1]
    d = data[:, nb_var:].reshape(N,1)
    
    return x, d, N, nb_var, nb_cible


def normalisation(x):
    """
    

    Parametres
    ----------
    x : matrice des données de dimension [N, nb_var]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Retour
    -------
    x_norm : matrice des données centrées-réduites de dimension [N, nb_var]
    mu : moyenne des variables de dimension [1,nb_var]
    sigma : écart-type des variables de dimension [1,nb_var]

    """
    
    mu = np.mean(x, 0)
    sigma = np.std(x, 0)
    x_norm = (x - mu) / sigma

    return x_norm, mu, sigma


def decoupage_donnees(x,d,prop_val=0.2, prop_test=0.2):
    """ Découpe les données initiales en trois sous-ensembles distincts d'apprentissage, de validation et de test
    
    Parametres
    ----------
    x : matrice des données de dimension [N, nb_var]
    d : matrice des valeurs cibles [N, nb_cible]
    prop_val : proportion des données de validation sur l'ensemble des données (entre 0 et 1)
    prop_test : proportion des données de test sur l'ensemble des données (entre 0 et 1)
    
    avec N : nombre d'éléments, nb_var : nombre de variables prédictives, nb_cible : nombre de variables cibles

    Retour
    -------
    x_app : matrice des données d'apprentissage
    d_app : matrice des valeurs cibles d'apprentissage
    x_val : matrice des données de validation
    d_val : matrice des valeurs cibles de validation
    x_test : matrice des données de test
    d_test : matrice des valeurs cibles de test

    """
    
    indice_app = int(x.shape[0]*(1-prop_test-prop_val)) #Je crée l'indice d'apprentissage en fonction du ratio
    indice_val = indice_app + int(x.shape[0]*prop_val) #Je crée l'indice de validation en fonction du ratio
    indice_test = indice_val + int(x.shape[0]*prop_test) #Je crée l'indice en fonction du ratio
    
    x_app, x_val, x_test = x[0:indice_app,:], x[indice_app:indice_val,:], x[indice_val:indice_test+1,:] #Je crée les 6 matrices en fonction des indices
    d_app, d_val, d_test = d[0:indice_app,:], d[indice_app:indice_val,:], d[indice_val:indice_test+1,:]
    
    return x_app, d_app, x_val, d_val, x_test, d_test


def lineaire(z, deriv=False):
    """ Calcule la valeur de la fonction linéaire ou de sa dérivée appliquée à z
    
    Parametres
    ----------
    z : peut être un scalaire ou une matrice
    deriv : booléen. Si False renvoie la valeur de la fonction linéire, si True renvoie sa dérivée


    Return
    -------
    s : valeur de la fonction linéaire appliquée à z ou de sa dérivée. Même dimension que z

    """
    if deriv:       
        return 1     
    else :
        return z
    

def sigmoide(z, deriv=False):
    """ Calcule la valeur de la fonction sigmoide ou de sa dérivée appliquée à z
    
    Parametres
    ----------
    z : peut être un scalaire ou une matrice
    deriv : booléen. Si False renvoie la valeur de la fonction sigmoide, si True renvoie sa dérivée

    Return
    -------
    s : valeur de la fonction sigmoide appliquée à z ou de sa dérivée. Même dimension que z

    """

    s = 1 / (1 + np.exp(-z))
    if deriv:
        return s * (1 - s)
    else :
        return s
    

def relu(z, deriv=False):
    """ Calcule la valeur de la fonction relu ou de sa dérivée appliquée à z
    
    Parametres
    ----------
    z : peut être un scalaire ou une matrice
    deriv : booléen. Si False renvoie la valeur de la fonction relu, si True renvoie sa dérivée

    Return
    -------
    s : valeur de la fonction relu appliquée à z ou de sa dérivée. Même dimension que z

    """

    r = np.zeros(z.shape)
    if deriv:
        pos = np.where(z>=0)
        r[pos] = 1.0
        return r
    else :    
        return np.maximum(r,z)
    

def calcule_cout_mse(y, d):
    """ Calcule la valeur de la fonction cout MSE (moyenne des différences au carré)
    
    Parametres
    ----------
    y : matrice des données prédites 
    d : matrice des données réelles 
    
    Return
    ------->
    cout : nombre correspondant à la valeur de la fonction cout (moyenne des différences au carré)

    """
    cout=0

    for i in range(len(y)) : 
        cout += (y[i]-d[0][i])**2
    return cout/2

def passe_avant(x, W, b, activation):
    """ Réalise une passe avant dans le réseau de neurones
    
    Parametres
    ----------
    x : matrice des entrées, de dimension nb_var x N
    W : liste contenant les matrices des poids du réseau
    b : liste contenant les matrices des biais du réseau
    activation : liste contenant les fonctions d'activation des couches du réseau

    avec N : nombre d'éléments, nb_var : nombre de variables prédictives 

    Return
    -------
    a : liste contenant les potentiels d'entrée des couches du réseau
    h : liste contenant les sorties des couches du réseau

    """

    h = [x]
    a = []
    for i in range(len(b)): 
        a_i = np.dot(W[i], h[i]) + b[i]
        h_i = activation[i](a_i, False)
        a.append(a_i)
        h.append(h_i)
    return a, h


def passe_arriere(delta_h, a, h, W, activation):
    """ Réalise une passe arrière dans le réseau de neurones (rétropropagation)
    
    Parametres
    ----------
    delta_h : matrice contenant les valeurs du gradient du coût par rapport à la sortie du réseau
    a : liste contenant les potentiels d'entrée des couches du réseau
    h : liste contenant les sorties des couches du réseau
    W : liste contenant les matrices des poids du réseau
    activation : liste contenant les fonctions d'activation des couches du réseau
    
    Return
    -------
    delta_W : liste contenant les matrice des gradients des poids des couches du réseau
    delta_b : liste contenant les matrice des gradients des biais des couches du réseau

    """

    delta_b = [np.zeros(k.shape) for k in b]
    delta_W = [np.zeros(w.shape) for w in W]
    
    for i in range(len(W)-1, -1, -1):
        delta = delta_h * activation[i](a[i], True)
        print(delta)
        delta_b[i]=delta
        delta_W[i]=np.dot(delta, h[i].T)
        delta_h = np.dot(W[i].T, delta)

    return delta_W, delta_b




# ===================== Partie 1: Lecture et normalisation des données =====================
print("Lecture des données ...")

x, d, N, nb_var, nb_cible = lecture_donnees("TD6-donnees/houses.txt")

# Affichage des 10 premiers exemples du dataset
print("Affichage des 2 premiers exemples du dataset : ")
for i in range(0, 2):
    print(f"x = {x[i,:]}, d = {d[i]}")
    
x_app, d_app, x_val, d_val, x_test, d_test =  decoupage_donnees(x,d)
    
# Normalisation des variables (centrage-réduction)
print("Normalisation des variables ...")
x, mu, sigma = normalisation(x)
dmax = d.max()
d = d / dmax

# Découpage des données en sous-ensemble d'apprentissage, de validation et de test
x_app, d_app, x_val, d_val, x_test, d_test =  decoupage_donnees(x,d)

# ===================== Partie 2: Apprentissage =====================

# Choix du taux d'apprentissage et du nombre d'itérations
alpha = 0.001
nb_iters = 2
couts_apprentissage = np.zeros(nb_iters)
couts_validation = np.zeros(nb_iters)

# Dimensions du réseau
D_c = [nb_var, 5, nb_cible] # liste contenant le nombre de neurones pour chaque couche 
activation = [relu, lineaire]

# Initialisation aléatoire des poids du réseau
W = []
b = []
for i in range(len(D_c)-1):    
    W.append(2 * np.random.random((D_c[i+1], D_c[i])) - 1)
    b.append(np.zeros((D_c[i+1],1)))

x_app = x_app.T # Les données sont présentées en entrée du réseau comme des vecteurs colonnes
d_app = d_app.T 

x_val = x_val.T # Les données sont présentées en entrée du réseau comme des vecteurs colonnes
d_val = d_val.T 

x_test = x_test.T # Les données sont présentées en entrée du réseau comme des vecteurs colonnes
d_test = d_test.T 
y_test = []

for t in range(nb_iters):
    
    y_app = []
    y_val = []
    
    for k in range(len(x_app[0])) : 
        ###############################################################################
        # Passe avant : calcul de la sortie prédite y sur les données d'apprentissage #
        ###############################################################################
        a, h = passe_avant(x_app[0][k], W, b, activation)
        y_app.append(h[-1]) # Sortie prédite

        ####################################
        # Passe arrière : rétropropagation #
        ####################################
        delta_h = (y_app[k]-d_app[0][k]) # Pour la dernière couche 
        delta_W, delta_b = passe_arriere(delta_h, a, h, W, activation)
      
        #############################################
        # Mise à jour des poids et des biais  #######
        ############################################# 
        for i in range(len(b)-1,-1,-1):
            b[i] -= alpha * delta_b[i]
            W[i] -= alpha * delta_W[i]
    
    for k in range(len(x_val[0])) : 
        
        #############################################################################
        # Passe avant : calcul de la sortie prédite y sur les données de validation #
        #############################################################################
        a, h = passe_avant(x_val[0][k], W, b, activation)
        y_val.append(h[-1]) # Sortie prédite
    
    ###########################################
    # Calcul de la fonction perte de type MSE #
    ###########################################
    couts_apprentissage[t] = calcule_cout_mse(y_app,d_app)
    couts_validation[t] = calcule_cout_mse(y_val,d_val)
    


print("Coût final sur l'ensemble d'apprentissage : ", couts_apprentissage[-1])
print("Coût final sur l'ensemble de validation : ", couts_validation[-1])

# Affichage de l'évolution de la fonction de cout lors de la rétropropagation
plt.figure(0)
plt.title("Evolution de le fonction de coût lors de la retropropagation")
plt.plot(np.arange(couts_apprentissage.size), couts_apprentissage, label="Apprentissage")
plt.plot(np.arange(couts_validation.size), couts_validation, label="Validation")
plt.legend(loc="upper left")
plt.xlabel("Nombre d'iterations")
plt.ylabel("Coût")
plt.show()

# ===================== Partie 3: Evaluation sur l'ensemble de test =====================

y_test=[]

for k in range(len(x_test[0])) : 
    #######################################################################
    # Passe avant : calcul de la sortie prédite y sur les données de test #
    #######################################################################
    a, h = passe_avant(x_test[0][k], W, b, activation)
    y_test.append(h[-1]) # Sortie prédite
    
cout = calcule_cout_mse(y_test,d_test)
print("Coût sur l'ensemble de test : ", cout[0][0])

y_c, x_c, d_c = [],[],[]

for k in range(len(y_val)) : 
    y_c.append(y_val[k][0][0])
    x_c.append(x_val[0][k])
    d_c.append(d_val[0][k])
    
    
plt.figure(1)
plt.grid()
plt.title("Courbe de régression")
plt.scatter(x_c, y_c, color='r')
plt.scatter(x_c, d_c, color='b')

#bon les courbes sont pas très convaincantes mais comme ça converge bien je sais pas quoi dire de plus
