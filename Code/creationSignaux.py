import numpy as np 
import code
import csv

"""
D1=np.array([[1,1,2,5,0,0,3,-2,1,2,2,2,5,1,3,1,-1,2,9,5,5,1,1,5],
[0,-1,4,2,-1,1,0,0,5,0,2,2,7,-12,2,5,5,2,7,4,-9,-2,1,2],
[1,3,1,1,5,1,2,2,1,1,1,1,5,0,-1,1,0,1,2,1,1,2,5,5],
[0,1,5,1,5,2,2,-2,5,0,-4,5,1,5,0,0,-1,-4,-8,2,2,-1,1,0],
[0,-1,2,3,2,2,3,1,1,0,0,0,0,4,-1,-2,0,7,4,3,4,-1,1,0],
[-1,8,6,3,2,2,2,4,-2,-3,-4,1,1,1,1,0,-2,-3,4,1,1,-1,1,0]])
X1=np.array([-10,-10,10,20,15,10])

X2=np.array([[-10],[10],[10],[20],[15],[10]])
"""
# Ouvrir le fichier csv
with open('DonneesCS22.csv', 'r') as f:
    # Créer un objet csv à partir du fichier
    obj = csv.reader(f)


# Ouvrir le fichier csv
with open('DonneesCS22.csv', 'r') as f:
	# Créer un objet csv à partir du fichier
	obj = csv.reader(f)
	i=0
	test=np.array([])
	for ligne in obj:
		if (i!=1):
			i+=1
		else:
			test=np.append(test,ligne)



test2=np.zeros((98,108))
k=0
for i in range(98):
	for j in range(108):
		test2[i,j]=test[k]
		k+=1

donnees_appr=test2
dicoinit=donnees_appr[:,0:100]

norms=np.linalg.norm(dicoinit,axis=0)
dicoinit=dicoinit/norms


[Dico,chapeauf,nbIter]=code.k_SVD(donnees_appr,dicoinit,0.01,100,90)

print(nbIter)


with open('DonneesCS222.csv', 'r') as f:
    # Créer un objet csv à partir du fichier
    obj = csv.reader(f)


# Ouvrir le fichier csv
with open('DonneesCS222.csv', 'r') as f:
	# Créer un objet csv à partir du fichier
	obj = csv.reader(f)
	i=0
	test=np.array([])
	for ligne in obj:
		if (i!=1):
			i+=1
		else:
			test=np.append(test,ligne)

k=0
donneesTest=np.zeros((98,3))
for i in range(98):
	for j in range(3):
		donneesTest[i,j]=test[k]
		k+=1

[parcimonie1,residu1,k1]=code.OMP(Dico,donneesTest[:,0],0.01,100)
print(np.linalg.norm(donnees_appr[:,101]-np.dot(Dico,chapeauf[:,101])))
