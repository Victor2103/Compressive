import math ;
import numpy as np;
import csv ; 

D1=np.array([[1,1,2,5,0,0,3,-2,1,2,2,2,5,1,3,1,-1,2,9,5,5,1,1,5],
[0,-1,4,2,-1,1,0,0,5,0,2,2,7,-12,2,5,5,2,7,4,-9,-2,1,2],
[1,3,1,1,5,1,2,2,1,1,1,1,5,0,-1,1,0,1,2,1,1,2,5,5],
[0,1,5,1,5,2,2,-2,5,0,-4,5,1,5,0,0,-1,-4,-8,2,2,-1,1,0],
[0,-1,2,3,2,2,3,1,1,0,0,0,0,4,-1,-2,0,7,4,3,4,-1,1,0],
[-1,8,6,3,2,2,2,4,-2,-3,-4,1,1,1,1,0,-2,-3,4,1,1,-1,1,0]])
X1=np.array([-10,-10,10,20,15,10])

X2=np.array([[-10],[10],[10],[20],[15],[10]])
print(np.shape(X2))
def OMP(D,X,eps,IterMax):
    R=X ; 
    indices=np.array([],dtype=int);  
    k=0 ; 
    while (np.linalg.norm(R)>eps) and (k<=IterMax):
        tmp3=np.array([]); 
        for j in range(0,np.shape(D)[1]):
            dj=D[:,j]
            tmp=abs(np.dot(np.transpose(dj),R)) 
            tmp2=np.linalg.norm(dj); 
            tmp3=np.append(tmp3,tmp/tmp2) 
        m=np.argmax(tmp3)
        indices=np.append(indices,m)
        if (k==5):
            print(indices)
            print(A)
        A=D[:,indices]
        alpha=np.dot(np.linalg.pinv(A),X)
        R=X-np.dot(A,alpha)
        k=k+1
    print(np.dot(A,alpha))
    parcimonieuse=np.zeros((np.shape(D)[1],1))
    j=0
    for i in indices:
        parcimonieuse[i,0]=alpha[j]
        j+=1
    print(np.dot(D,parcimonieuse))



    print("Représentation parcimonieuse : ",parcimonieuse)
    print("Résidu : ",R)
    print("Nombre d'itérations : ",k)
    return(parcimonieuse,R,k)

[delta,Rf,kf]=OMP(D1,X2,0.0001,1000) 



def recherche_non_neg(delta,i):
    for j in range(0,delta.shape[1]):
        if (delta[i,j]!=0):
            wi=np.append(wi,j)
    return(wi)

tab=[]
tab2=np.array([])
f=open("DonneesProjet.csv") 
reader=csv.reader(f)
i=0;  
for row in reader :
    tab.append(row)

tab2=np.ones((98,216))

for i in range(98):
    for j in range(216):
        if (tab[i][j]!=''):
            tab2[i,j]=float(tab[i][j])

print("longeur =  :",tab2.shape)

def k_SVD(X,D,eps,N,k):
    [delta,Rf,kf]=OMP(D,X,eps,N)
    n=0
    while (np.linalg.norm(X-np.dot(D,delta))>eps) and (n<N):
        for i in range(1,k):
            Ei=X-np.dot(D,delta)+np.dot(D[:,i],delta[i,:])
            wi=recherche_non_neg(delta,i)
            C=np.identity(X.shape[1])
            omega_i=C[:,wi]
            if (omega_i==0):
                i_eme_colonne=X[:,i]
                norms=np.linalg.norm(i_eme_colonne,axis=2)
                D[:,i]=i_eme_colonne/norms
            else:
                Eir=np.dot(Ei,omega_i)
                [U,sigma,V]=np.linalg.svd(Eir)
                D[:,i]=U[:,1]
        [delta,Rf,kf]=OMP(D,X,eps,N)
        n=n+1
    print("dico final : ",D)
    print("Représentation parcimonieuse : ",delta)
    print("Nombre itérations : ",n)
    return(D,delta,n)

#[Df,deltaf,nf]=k_SVD(X1,D1,0.001,1000,4)


    

