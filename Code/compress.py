from cProfile import label
import math,cmath
import matplotlib.pyplot as plt
import numpy as np

import code

N=500

C=code.creationDCT(500)

X1=code.creationSignal(500)

alphaC=np.dot(np.transpose(C),X1)

print("Pour un signal de taille N=500 : ")

print("Précision avec la DCT et le dictionnaire direct :"+str(np.linalg.norm(X1-np.dot(C,alphaC))))

F=code.creationDFT(500)

alphaF=np.dot(np.transpose(F.conjugate()),X1)

print("Précision avec la DFT et le dictionnaire direct : "+str(np.linalg.norm(X1-np.dot(F,alphaF))))

t=[i/400 for i in range(150)]
plt.plot(t,X1[0:150,0],label="Signal d'origine")
plt.plot(t,np.dot(C,alphaC)[0:150,0],label="Représentation avec la base de cosinus discret")
plt.plot(t,np.dot(F,alphaF)[0:150,0],label="Signal avec la base de Fourier")
plt.legend(loc=2, prop={'size':10})
plt.show()
plt.close()

[alphaOMPC,RésiduOMPC,iterationOMPC]=code.OMP(C,X1,0.01,100)

print()
print("Précision avec la DCT et le dictionnaire obtenu avec OMP : "+str(np.linalg.norm(X1-np.dot(C,alphaOMPC)))
+" et Nombre d'itérations : "+str(iterationOMPC))

[alphaOMPF,residuOMPF,iterationOMPF]=code.OMP(F,X1,0.01,100)

print("Precision avec la DFT et le dictionnaire obtenu avec OMP : "+str(np.linalg.norm(X1-np.dot(F,alphaOMPF)))
+" et Nombre d'itérations : "+str(iterationOMPF))

X2=code.creationSignal(100)

print("\n")
print("Pour un signal de taille 100: ")

C=code.creationDCT(100)
alphaC=np.dot(np.transpose(C),X2)
print("Précision avec la DCT et le dictionnaire direct :"+str(np.linalg.norm(X2-np.dot(C,alphaC))))
F=code.creationDFT(100)
alphaF=np.dot(np.transpose(F),X2)
print("Précision avec la DFT et le dictionnaire direct : "+str(np.linalg.norm(X2-np.dot(F,alphaF))))

[alphaOMPC,RésiduOMPC,iterationOMPC]=code.OMP(C,X2,0.01,100)
print()
print("Précision avec la DCT et le dictionnaire obtenu avec OMP : "+str(np.linalg.norm(X2-np.dot(C,alphaOMPC)))
+" et Nombre d'itérations : "+str(iterationOMPC))
[alphaOMPF,residuOMPF,iterationOMPF]=code.OMP(F,X2,0.01,100)
print("Precision avec la DFT et le dictionnaire obtenu avec OMP : "+str(np.linalg.norm(X2-np.dot(F,alphaOMPF)))
+" et Nombre d'itérations : "+str(iterationOMPF))

print(np.shape(X2))
[test,nbiter]=code.IRLS(C,X2,0.1,100,2)

print(np.linalg.norm(X2-np.dot(C,test)))
print(nbiter)