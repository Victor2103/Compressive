import math ;
import numpy as np;




def OMP(D,X,eps,IterMax):
    R=X ; 
    indices=np.array([],dtype=int);  
    k=0 ; 
    while (np.linalg.norm(R)>eps) and (k<IterMax):
        tmp3=np.array([]); 
        for j in range(0,np.shape(D)[1]):
            dj=D[:,j]
            tmp=abs(np.dot(np.transpose(dj),R)) 
            tmp2=np.linalg.norm(dj); 
            tmp3=np.append(tmp3,tmp/tmp2) 
        m=np.argmax(tmp3)
        indices=np.append(indices,m)
        A=D[:,indices]
        alpha=np.dot(np.linalg.pinv(A),X)
        R=X-np.dot(A,alpha)
        k=k+1
    parcimonieuse=np.zeros((np.shape(D)[1],1))
    j=0
    for i in indices:
        parcimonieuse[i,0]=alpha[j]
        j+=1
    return(parcimonieuse,R,k)



def recherche_non_neg(delta,i):
    for j in range(0,delta.shape[1]):
        if (delta[i,j]!=0):
            wi=np.append(wi,j)
    return(wi)


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




    

