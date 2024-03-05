import numpy as np
import cv2
from matplotlib import pyplot as plt
from operators import *
from metrics import *


# INITIALISATION
# génération d'une vérité terrain 1D xbar
N = 100
dom  = np.arange(1,N+1)
xbar = np.zeros((N,),float)
xbar[25:50] =  np.sin(.25*dom[25:50])
xbar[60:70] = -1.
xbar[75:90] =  1.


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
    return np.linalg.norm(x-z, ord=1)

# fonction de coût globale
def E(x,lam):
    return f(x) + lam * R(x)


# fonction prox norme 1
def prox(xn,gamma):
    x = np.zeros((len(xn),),float) * np.nan
    for i in range(len(xn)):
        if xn[i] >= gamma:
            x[i] = xn[i] - gamma
        elif xn[i] <= -gamma:
            x[i] = xn[i] + gamma
        else:
            x[i] = 0
    return x


# ALGORITHME
# paramètres du modèle
# lam = 5.5
# lip = 2*opNorm(H,Ht,'1D')**2 + 2*lam*opNorm(D,Dt,'1D')**2
# taux = 0.9/lip
gamma = 0.51
taux = 0.1
print(taux)

# paramètres de l'algo
Niter = 1000                                 # nombre max d'itérations

# initialisation des variables
En = np.zeros((Niter,),float) * np.nan
xn = np.random.randn(len(xbar))									# x0

# itérations
for i in range(1,Niter):
    xn = prox(xn - 2*gamma*(xn-z),taux)

    En[i] = E(xn,taux)
    
xhat = xn

print(mse(xn,xbar))
print(snr(xn,xbar))
# AFFICHAGE DU RÉSULTAT
plt.figure()

plt.subplot(211)
plt.plot(dom,xbar,':',color='black',label='xbar')
plt.plot(dom,z,'-',color='orange',label='z')
plt.plot(dom,xhat,'--',color='crimson',label='xhat')
plt.figlegend()
plt.title('signaux')

plt.subplot(212)
plt.loglog(En)
plt.title('fonction de coût')
plt.show()