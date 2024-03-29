import numpy as np
import cv2
from matplotlib import pyplot as plt
from operators import *
from metrics import *


# INITIALISATION
# génération d'une vérité terrain 1D xbar
N = 100
xbar = cv2.imread('cameraman.tif', 0)/255.0


# CRÉATION DE LA DONNÉE DÉGRADÉE
# choix de l'opérateur d'acquisition
h = generatePSF('1D','gaussian',3)
def H(x):
    return A(x,h)
def Ht(x):
    return At(x,h)

# création de la donnée z, dégradée par un bruit blanc gaussien d'écart-type sig
sig = 0.1
z = xbar + sig*np.random.standard_normal((xbar.shape))


# FONCTION DE COÛT
# attache aux données
def f(x):
    return np.linalg.norm(x-z)**2

# régularisation
def R(x):
    return np.linalg.norm(x, ord=1)

# fonction de coût globale
def E(x,lam):
    return f(x) + lam * R(x)


# fonction prox norme 1
def prox(x,gamma,lam):
    return np.maximum(x-lam*gamma , np.minimum(0 , x+lam*gamma))
    # retourne la fonction du prox à l'aide du min-max


# ALGORITHME
# paramètres du modèle
lam = 0.01
# lip = 2*opNorm(H,Ht,'1D')**2 + 2*lam*opNorm(D,Dt,'1D')**2
# taux = 0.9/lip
gamma = 0.4
print(gamma)

# paramètres de l'algo
Niter = 1000                                 # nombre max d'itérations

# initialisation des variables
En = np.zeros((Niter,),float) * np.nan
xn = np.random.randn(len(xbar),len(xbar))									# x0

# itérations
for i in range(1,Niter):
    xn = prox(xn - 2*gamma*(xn-z),gamma,lam)

    En[i] = E(xn,lam)
    
xhat = xn

print(mse(xn,xbar))
print(snr(xn,xbar))
# AFFICHAGE DU RÉSULTAT
plt.figure()

plt.subplot(221)
plt.imshow(xbar, cmap='gray')
plt.title('Image originelle')
plt.axis('off')

plt.subplot(222)
plt.imshow(z,cmap='gray')
plt.title('Image bruitée')
plt.axis('off')

plt.subplot(223)
plt.imshow(xhat,cmap='gray')
plt.title('Image reconstruite')
plt.axis('off')

plt.subplot(224)
plt.loglog(En)
plt.title('fonction de coût')
plt.show()