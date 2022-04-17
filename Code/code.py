import math
import numpy as np;




def OMP(D,X,eps,IterMax):
    R=X ; 
    indices=np.array([],dtype=int);  
    k=0 ; 
    while (np.linalg.norm(R)>eps) and (k<IterMax):
        tmp3=np.array([]); 
        for j in range(0,np.shape(D)[1]):
            dj=D[:,j]
            tmp=abs(np.dot(np.transpose(dj.conjugate()),R)) 
            tmp2=np.linalg.norm(dj); 
            tmp3=np.append(tmp3,tmp/tmp2) 
        m=np.argmax(tmp3)
        indices=np.append(indices,m)
        A=D[:,indices]
        alpha=np.dot(np.linalg.pinv(A),X)
        R=X-np.dot(A,alpha)
        k=k+1
    parcimonieuse=np.zeros((np.shape(D)[1],1),dtype='complex')
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
    chapeau=np.array(delta,dtype='complex')
    for i in range(np.shape(D)[0]-1):
        chapeau=np.concatenate((chapeau,delta),axis=1)
    print(np.shape(chapeau))
    print(np.shape(D))
    while (np.linalg.norm(X-np.dot(D,delta))>eps) and (n<N):
        for i in range(1,k):
            Ei=X-np.dot(D,delta)+np.dot(D[:,i],chapeau[i,:])
            print("une fois")
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


def creationSignal(taille):
	fe=400
	t=[i/fe for i in range(taille)]

	f0=50
	sinus=np.zeros(taille)
	for i in range(taille):
		sinus[i]=np.sin(2*np.pi*f0*t[i])

	fp=100
	tporteuse=[i/fe for i in range(taille)]
	porteuse=np.zeros(taille)
	for i in range(taille):
		porteuse[i]=np.cos(2*np.pi*fp*tporteuse[i])
	
	X=np.zeros((taille,1))
	for i in range(0,len(porteuse)):
		X[i,0]=sinus[i]*porteuse[i]
	return(X)

def creationDFT(taille):
	F=np.zeros((taille,taille),dtype='complex')
	for i in range(taille):
		for j in range(taille):
			tmp=-2*math.pi*i*j/taille
			F[i,j]=1/math.sqrt(taille)*complex(math.cos(tmp),math.sin(tmp))
	return(F)

def creationDCT(taille):
	C=np.zeros((taille,taille))
	for k in range(taille):
		for p in range(taille):
			if (k==0):
				C[k,p]=1/np.sqrt(taille)
			else :
				C[k,p]=(np.sqrt(2/taille))*np.cos(np.pi*(2*k+1)*p/(2*taille))
	return(C)

def IRLS(D,X,eps,Itermax,p):
    inverse=np.linalg.inv(np.dot(D,np.transpose(D)))
    alpha=np.dot(np.transpose(D),inverse)
    alpha=np.dot(alpha,X)
    print(np.shape(alpha))
    Omega=np.zeros((np.shape(X)[0],np.shape(X)[0]))
    k=0
    wi=[]
    for i in range(np.shape(X)[0]):
            wi.append(math.pow(math.pow(abs(alpha[i]),2)+eps,p/2-1))
            Omega[i,i]=wi[i]
    Q=np.dot(Omega,np.transpose(Omega))
    qDt=np.dot(Q,np.transpose(D))
    inv=np.linalg.inv(np.dot(D,qDt))
    tmp=np.dot(qDt,inv)
    alpha2=np.dot(tmp,X)
    k=k+1
    while (abs(np.linalg.norm(alpha)-np.linalg.norm(alpha2))>math.sqrt(eps)/100)and(k<Itermax):
        alpha=alpha2
        wi=[]
        for i in range(np.shape(X)[0]):
            wi[i]=math.pow(math.pow(abs(alpha[i]),2)+eps,p/2-1)
            Omega[i,i]=wi[i]
        Q=np.dot(Omega,np.transpose(Omega))
        qDt=np.dot(Q,np.transpose(D))
        inv=np.linalg.inv(np.dot(D,qDt))
        tmp=np.dot(qDt,inv)
        alpha2=np.dot(tmp,X)
        if (abs(np.linalg.norm(alpha)-np.linalg.norm(alpha2))<math.sqrt(eps)/100)and(eps>math.pow(1/10,8)):
            eps=eps/10
        k=k+1
    return(alpha2,k)
    
    

    

