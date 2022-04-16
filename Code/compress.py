from cProfile import label
import math,cmath
import matplotlib.pyplot as plt
import numpy as np

import code

N=500
C=np.zeros((N,N))
for k in range(N):
	for p in range(N):
		if (k==0):
			C[k,p]=1/np.sqrt(N)
		else :
			C[k,p]=(np.sqrt(2/N))*np.cos(np.pi*(2*k+1)*p/(2*N))
		


#Echantillonage
fe=400
t=[i/fe for i in range(N)]

f0=50
sinus=np.zeros(N)
for i in range(N):
	sinus[i]=np.sin(2*np.pi*f0*t[i])

fp=100
tporteuse=[i/fe for i in range(N)]
porteuse=np.zeros(N)
for i in range(N):
	porteuse[i]=np.cos(2*np.pi*fp*tporteuse[i])
	
X=np.zeros((500,1))
for i in range(0,len(porteuse)):
	X[i,0]=sinus[i]*porteuse[i]
#Pb dimension
alpha=np.zeros((500,1))

alphaC=np.dot(np.transpose(C),X)


print(np.linalg.norm(X-np.dot(C,alphaC)))


F=np.zeros((N,N),dtype='complex')
for i in range(N):
	for j in range(N):
		tmp=-2*math.pi*i*j/N
		F[i,j]=1/math.sqrt(N)*complex(math.cos(tmp),math.sin(tmp))



alphaF=np.dot(np.transpose(F),X)

print(np.linalg.norm(X-np.dot(F,alphaF)))

"""
plt.plot(t,X,label="Signal d'origine")
plt.plot(t,np.dot(C,alphaC),label="Représentation avec la base de cosinus discret")
plt.plot(t,np.dot(F,alphaF),label="Signal avec la base de Fourier")
plt.legend(loc=2, prop={'size':5})
plt.show()
plt.close()
"""

[alphaOMPC,RésiduOMPC,iteration]=code.OMP(C,X,0.01,100)

print("Précision entre les 2 représentations :"+str(np.linalg.norm(alphaC-alphaOMPC)))
