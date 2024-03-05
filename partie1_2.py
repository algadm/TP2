import numpy as np
import cv2
from matplotlib import pyplot as plt
from operators import *
from metrics import *

# fonction sous-gradient
def grad(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return np.random.random()*2 - 1
    
# fonction Moreau
def moreau(xn,gamma):
    x = np.zeros((len(xn),),float) * np.nan
    for i in range(len(xn)):
        if xn[i] >= gamma:
            x[i] =  xn[i] - gamma/2
        elif xn[i] < gamma and xn[i] > -gamma:
            x[i] = xn[i]**2 / (2*gamma)
        else:
            x[i] =  -xn[i] - gamma/2
    return x

# fonction prox norme 1
def prox(x,gamma):
    return x - gamma if x >= gamma else x + gamma if x <= -gamma else 0
    
Niter = 1000
taux = 0.2
Nmax = 3
dom  = np.arange(-Nmax,Nmax+1)
x = np.random.random()*2*Nmax - Nmax
En = np.zeros((Niter,),float) * np.nan
vec = np.linspace(-Nmax,Nmax)

# FONCTION DE COÛT
# attache aux données
def f(x):
    return abs(x)

def E(x):
    return f(x)

# AFFICHAGE DU RÉSULTAT
plt.figure()

plt.subplot(211)
# itérations
for i in range(1,Niter):
    plt.plot(x,abs(x),'*',color='red')
    x = prox(x,taux)
    En[i] = E(x)


plt.plot(dom,abs(dom),':',color='black',label='xbar')
plt.plot(vec,moreau(vec,taux),'-',color='black')

plt.figlegend()
plt.title('signaux')

plt.subplot(212)
plt.loglog(En)
# plt.plot(np.linspace(1, Niter, len(En)), En)
plt.title('fonction de coût')

plt.show()