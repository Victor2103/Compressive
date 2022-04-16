import matplotlib.pyplot as plt
import numpy as np

N=500
C=np.zeros((N,N))
for k in range(N):
	for p in range(N):
		if (p==0):
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

alpha=np.dot(np.transpose(C),X)


print(np.linalg.norm(X-np.dot(C,alpha)))

