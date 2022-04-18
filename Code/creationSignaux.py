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
	print(obj)
	i=0
	test=np.array([])
	for ligne in obj:
		if (i!=1):
			i+=1
		else:
			test=np.append(test,ligne)


print(10584/108)
test2=np.zeros((98,108))
k=0
for i in range(98):
	for j in range(108):
		test2[i,j]=test[k]
		k+=1

donnees_appr=test2

[Dico,chapeauf,nbIter]=code.k_SVD(test2,dicoinit,0.01,100,100)