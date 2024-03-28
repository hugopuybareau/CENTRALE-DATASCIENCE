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
    X : matrice des données de dimension [N, nb_var]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    """

    data = np.loadtxt(nom_fichier, delimiter=delimiteur)
    nb_var = data.shape[1] - 1
    N = data.shape[0]

    X = data[:, :-1]
    Y = data[:, -1].reshape(N,1)
    
    return X, Y, N, nb_var   

def normalisation(X):
    """
    

    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Retour
    -------
    X_norm : matrice des données centrées-réduites de dimension [N, nb_var]
    mu : moyenne des variables de dimension [1,nb_var]
    sigma : écart-type des variables de dimension [1,nb_var]

    """

    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mu) / sigma  

    return X_norm, mu, sigma

def sigmoide(z):
    """ Calcule la valeur de la fonction sigmoide appliquée à z
    
    Parametres
    ----------
    
    z : peut être un scalaire ou une matrice
    Return
    -------
    s : valeur de sigmoide de z. Même dimension que z

    """
    s = 1 / (1 + np.exp(-z))
    
    return s

def calcule_cout(X, Y, theta):
    
    """ Calcule la valeur de la fonction cout (moyenne des différences au carré)
    
    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Return
    -------
    cout : nombre correspondant à la valeur de la fonction cout (moyenne des différences au carré)

    """
    cout = - np.sum(Y*np.log(sigmoide(X.dot(theta.T)))+(1-Y)*np.log(1-sigmoide(X.dot(theta.T))))    #J(theta) = - l(theta) puis formule page 10 du poly avec ftheta = g(theta.TX) = sigmoide(theta.TX)

    return cout

def descente_gradient(X, Y, theta, alpha, nb_iters):
    """ Apprentissage des parametres de regression linéaire par descente du gradient
    
    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    alpha : taux d'apprentissage
    nb_iters : nombre d'itérations
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives


    Retour
    -------
    theta : matrices contenant les paramètres theta appris par descente du gradient de dimension [1, nb_var+1]
    J_history : tableau contenant les valeurs de la fonction cout pour chaque iteration de dimension nb_iters


    """
    
    # Initialisation de variables utiles
    N = X.shape[0]
    J_history = np.zeros(nb_iters)

    for i in range(0, nb_iters):

        error = sigmoide(X.dot(theta.T)) - Y
        theta -= (alpha/N)*np.sum(X*error, 0) #formule page 15 du poly car expression identique à celle de la régression linéaire mais avec f(theta) différente (sigmoïde)" voir page 12

        J_history[i] = calcule_cout(X, Y, theta) 

    return theta, J_history

def prediction(X,theta):
    """ Predit la classe de chaque élement de X
    
    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives


    Retour
    -------
    p : matrices de dimension [N,1] indiquant la classe de chaque élement de X (soit 0, soit 1)

    """
    N = X.shape[0] #initialisation des variables utiles
    p = np.zeros((N,1)) #initialisation de p et évite de faire un else dans mon for 
    
    for i in range(N) :
        if X.dot(theta.T)[i] >= 0.5 :
            p[i] = 1
    return p

def taux_classification(Ypred,Y):
    """ Calcule le taux de classification (proportion d'éléments bien classés)
    
    Paramètres
    ----------
    Ypred : matrice contenant les valeurs de classe prédites de dimension [N, 1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    
    avec N : nombre d'éléments 


    Retour
    -------
    t : taux de classification (valeur scalaire)

    """

    t = 0 #initialisation de t
    N=Y.shape[0]
    
    for i in range(N):
        if Ypred[i] == Y[i] :
            t+=1
            
    t=(t/N)*100 #on ramène t à un pourcentage
    return t
    
def affichage(X, Y):
    """ Affichage en 2 dimensions des données (2 dimensions de X) et représentation de la 
        classe (indiquée par Y) par une couleur
    

    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Retour
    -------
    None

    """
    plt.grid()

    for i in range(len(Y)):
        if Y[i] == 1:
            plt.scatter(X[i, 2], X[i, 1], c='g', marker='x')
        elif Y[i] == 0:
            plt.scatter(X[i, 2], X[i, 1], c='r', marker='x')

    plt.show()
    
# ===================== Partie 1: Lecture et normalisation des données=====================
print("Lecture des données ...")

X, Y, N, nb_var = lecture_donnees("TD5-donnees/notes.txt")

# Affichage des 10 premiers exemples du dataset
print("Affichage des 10 premiers exemples du dataset : ")
for i in range(0, 10):
    print(f"x = {X[i,:]}, y = {Y[i]}")
    
# Normalisation des variables (centrage-réduction)
print("Normalisation des variables ...")

X, mu, sigma = normalisation(X)

# Ajout d'une colonne de 1 à X (pour theta0)
X = np.hstack((np.ones((N,1)), X)) 

# Affichage des points en 2D et représentation de leur classe réelle par une couleur
if nb_var == 2 :
    plt.figure(0)
    plt.title("Disposition des points en 2D")
    affichage(X,Y)

# ===================== Partie 2: Descente du gradient =====================
print("Apprentissage par descente du gradient ...")

# Choix du taux d'apprentissage et du nombre d'itérations
alpha = 0.05
nb_iters = 50000

# Initialisation de theta et réalisation de la descente du gradient
theta = np.zeros((1,nb_var+1))
theta, J_history = descente_gradient(X, Y, theta, alpha, nb_iters)

# Affichage de l'évolution de la fonction de cout lors de la descente du gradient
plt.figure(1)
plt.title("Evolution de le fonction de cout lors de la descente du gradient")
plt.plot(np.arange(J_history.size), J_history)
plt.xlabel("Nombre d'iterations")
plt.ylabel("Cout J")

# Affichage de la valeur de theta
print(f"Theta calculé par la descente du gradient : {theta}")

# Evaluation du modèle
Ypred = prediction(X,theta)

print("Taux de classification : ", taux_classification(Ypred,Y),"%") #J'ajoute ça puisque j'ai défini t comme un pourcentage

# Affichage des points en 2D et représentation de leur classe prédite par une couleur
if nb_var == 2 :
    plt.figure(2)
    plt.title("Disposition des points en 2D après test")
    affichage(X,Ypred) #On obtient quelque chose de logique, les points verts et les points rouges sont séparés par la droite du 50 de moyenne ce qui est logique puisque la couleur est définie en fonction de la capacité du candidat à valider, soit sa capacité à avoir des notes supérieures à la moyenne
    #et les nuages se ressemblent plutot bien c'est satisfaisant
plt.show()

print("Regression logistique Terminée.")