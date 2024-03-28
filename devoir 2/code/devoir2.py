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
    
def train_test(X, Y, ratio): 
    """ 
    
    Cette fonction sépare le jeu de données principale en 2 jeu de données,
    un pour l'entrainement et un pour le test du modèle
    
    """
    
    indice_train = int(X.shape[0]*ratio) #Je crée l'indice en fonction du ratio
    X_train, X_test, Y_train, Y_test = X[0:indice_train,:] ,X[indice_train:,:],Y[0:indice_train, :], Y[indice_train:,:]
    
    return X_train, X_test, Y_train, Y_test

def diff_prediction(Y, classe) :
    
    """ 
        Je crée la matrice Y d'entrainements pour pouvoir faire le un contre tous
    """
    
    N = Y.shape[0] #initialisation des variables utiles
    Y_1ct = np.zeros((N,1))
    for i in range(N) : 
        if Y[i] == classe : 
            Y_1ct[i]=1
            
    return Y_1ct


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
    """ Apprentissage des parametres de regression logistique par descente du gradient
    
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

def prediction(X,mat_comp):
    """ Predit la classe de chaque élement de X
    
    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives


    Retour
    -------
    p : matrices de dimension [N,1] indiquant la classe de chaque élement de X (soit 0, soit 1, soit 2)

    """
    N = X.shape[0] #initialisation des variables utiles
    p = np.zeros((N,1)) #initialisation de p
    
    for i in range(N) :
        for k in range(3) : 
            if max(mat_comp[i]) == mat_comp[i][k] :
                p[i] = k 
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
    
def affichage(X_abs, X_ord, Y):
    
    """Affichage en 2 dimensions des données (2 dimensions de X) et représentation de la 
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

    plt.grid(True)
    for i in range(len(Y)):
        if Y[i] == 0:
            plt.scatter(X_abs[i], X_ord[i], c='g', marker='x') 
        elif Y[i] == 1:
            plt.scatter(X_abs[i], X_ord[i], c='r', marker='.')
        elif Y[i] == 2 : 
            plt.scatter(X_abs[i], X_ord[i], c='b', marker='o')

    plt.show()
    
# ===================== Partie 1: Lecture et normalisation des données=====================
print("Lecture des données ...")

X, Y, N, nb_var = lecture_donnees("TD5-donnees/iris.txt")    

#Je m'excuse à l'avance car j'ai pris le premier jeu de données du site que vous avez donné. Il est possible que vous le connaissiez déjà. 

# Affichage des 10 premiers exemples du dataset
print("Affichage des 10 premiers exemples du dataset : ")
for i in range(-2, 10):
    print(f"x = {X[i,:]}, y = {Y[i]}")
    
# Normalisation des variables (centrage-réduction)
print("Normalisation des variables ...")

X, mu, sigma = normalisation(X)

for i in range(-2, 10):
    print(f"x = {X[i,:]}")

# Ajout d'une colonne de 1 à X (pour theta0)
X = np.hstack((np.ones((N,1)), X)) 

# Séparation des données de test et des données d'entrainement en fonction d'un ratio à choisir

print("Séparation des données pour test et entrainement ...")
ratio = 0.2
X_train, X_test, Y_train, Y_test = train_test(X,Y, ratio)

print("On réalise l'entraînement sur", int(X.shape[0]*ratio), "valeurs")

#Affichage des points en 2D et représentation de leur classe réelle par une couleur
#if nb_var == 2 :
    #plt.figure(0)
    #plt.title("Disposition des points en 2D")
    #affichage(X,Y)

# ===================== Partie 2: Descente du gradient =====================
print("Apprentissage par descente du gradient ...")

# Choix du taux d'apprentissage et du nombre d'itérations
alpha = 0.01
nb_iters = 10000

# Initialisation de theta et réalisation de la descente du gradient
# Il y a 3 classes doncon fait 3 matrices de theta 
theta0, theta1, theta2 = np.zeros((1,nb_var+1)), np.zeros((1,nb_var+1)), np.zeros((1,nb_var+1))
Y_t0, Y_t1, Y_t2 = diff_prediction(Y_train, 0), diff_prediction(Y_train, 1), diff_prediction(Y_train, 2)
theta0, J_history0 = descente_gradient(X_train, Y_t0, theta0, alpha, nb_iters)
theta1, J_history1 = descente_gradient(X_train, Y_t1, theta1, alpha, nb_iters)
theta2, J_history2 = descente_gradient(X_train, Y_t2, theta2, alpha, nb_iters)

# Affichage de l'évolution de la fonction de cout lors de la descente du gradient
plt.figure(0)
plt.grid(True)
plt.title("Evolution de le fonction de cout lors de la descente du gradient")
plt.plot(np.arange(J_history0.size), J_history0, 'r')
plt.plot(np.arange(J_history1.size), J_history1, 'g')
plt.plot(np.arange(J_history2.size), J_history2, 'b')
plt.xlabel("Nombre d'iterations")
plt.ylabel("Couts J pour les 3 classes")


# Affichage de la valeur de theta
print("Affichage des 3 matrices theta : ")
print("Theta 0 =", theta0)
print("Theta 1 =", theta1)
print("Theta 2 =", theta2)

# Evaluation du modèle

mat_comp_train = np.concatenate((X_train.dot(theta0.T),X_train.dot(theta1.T), X_train.dot(theta2.T)), axis = 1) #Je crée la matrice de comparaison des contributions avant d'éxecuter la fonction de prédiction
print("La matrice de comparaison utilisée pour la prédiction est la suivante : ", mat_comp_train)

Ypred_train = prediction(X_train, mat_comp_train)

print("On obtient alors la matrice de prédiction suivante : ", Ypred_train)

print("Taux de classification pour l'entraînement : ", taux_classification(Ypred_train,Y_train),"%") #J'ajoute ça puisque j'ai défini t comme un pourcentage

if taux_classification(Ypred_train, Y_train) < 75 :
    print("Il est nécessaire de changer les valeurs d'alpha et le nombre d'itérations et de recalculer les matrices")
    
    #Je pourrai faire un while sur le taux de classification et modifier les valeurs de alpha et du nombre d'itérations pour rendre le modèle totalement autonome mais bon...
# Affichage des points en 2D et représentation de leur classe prédite par une couleur
#if nb_var == 2 :
    #plt.figure(2)
    #plt.title("Disposition des points en 2D")
    #affichage(X,Ypred) #On obtient quelque chose de logique, les points verts et les points rouges sont séparés par la droite du 50 de moyenne ce qui est logique puisque la couleur est définie en fonction de la capacité du candidat à valider, soit sa capacité à avoir des notes supérieures à la moyenne
    
#plt.show()

print("Entraînement du modèle de regression logistique terminé.")

#Réalisation du test pour le reste des valeurs

mat_comp_test = np.concatenate((X_test.dot(theta0.T),X_test.dot(theta1.T), X_test.dot(theta2.T)), axis = 1)
Ypred_test = prediction(X_test, mat_comp_test)
print("On obtient la matrice de prédiction suivante :", Ypred_test)
print("Taux de classification pour le test : ", taux_classification(Ypred_test, Y_test), "%")

print("Regression logistique terminée")

# ===================== Partie 3: Analyse =====================

plt.figure(1)
plt.title("Représentation des classes en fonction des sépales")
affichage(X_test[:,1], X_test[:, 2], Ypred_test)
plt.show()


plt.figure(2)
plt.title("Représentation des classes en fonction des pétales")
affichage(X_test[:,3], X_test[:, 4], Ypred_test)
plt.show()



