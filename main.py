from numba import jit
import numpy as np
import time
 
#measuring time
tau = time.time() 

#matrices for elliptic and hyperbolic behaviour
Mh=np.array([[2.0,1],[3,2]])
Me=np.array([[0.0,1],[0,-1]])

pospesi=True # @jit(nopython=pospesi)  <- activating JustInTime compilation for faster computing
    
def ft(N,psi):
    psi2=np.zeros([N])+0j
    premik=int((N-1)/2)
    for i in range(-premik,premik+1):
        for j in range(-premik,premik+1):
            psi2[i+premik]+=1/np.sqrt(N)*np.exp(-2*np.pi*1j*j*i/N)*psi[j+premik]
    return psi2

def ift(N,psi):
    psi2=np.zeros([N])+0j
    premik=int((N-1)/2)
    for i in range(-premik,premik+1):
        for j in range(-premik,premik+1):
            psi2[i+premik]+=1/np.sqrt(N)*np.exp(2*np.pi*1j*j*i/N)*psi[j+premik]
    return psi2

@jit(nopython=pospesi)  #returns norm of psi
def norm (psi):
    return np.sum(psi*np.conj(psi))

@jit(nopython=pospesi)  
def normiraj(psi):  #returns psi normed
    return psi/np.sqrt(norm(psi))
    
@jit(nopython=pospesi)     
def gauss(x,avg,sigma): #normal gauss function
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-avg)**2/(2*sigma**2))
  
@jit(nopython=pospesi)  
def get_psi_gauss(N,q0,p0,Q,P): #gauss state for quantum system
    q=np.zeros(N,dtype=np.complex_)
    premik=int((N-1)/2)
    sigma=1.0
    for i in range(-premik,premik+1):
        for j in range(-premik,premik+1):
            q[i+premik]+=np.exp(1j*2*np.pi*p0*float(i))*np.exp(-(np.pi*sigma/float(N))*(float(i)-q0*float(N)+float(j)*N)**2)*np.exp(1j*2*np.pi*p0*float(j)*float(N))
    q=normiraj(q)
    return q
    

@jit(nopython=pospesi)
def fat_delta(N,l):  
    l=float(l)
    l=np.abs(l)
    if(np.abs(l)<10**(-10)):
        return 1.0
    return np.sin(np.pi*l/2.0)/(np.sin(np.pi*l/(2.0*N))*N)

@jit(nopython=pospesi)   
def cycle(N,a):   #cycle integers back onto torus
    a=int(a)
    premik=int((N-1)/2)
    while a<-premik:
        a+=N
    while a>premik:
        a-=N
    return a+premik
    

@jit(nopython=pospesi)    
def get_wigner(N,psi):  #wigner function for NxN state, which is not coupled yet
    premik=int((N-1)/2)
    w=np.zeros((N,N),dtype=np.complex_)
    for n in range(-premik, premik+1):
        for k in range(-premik, premik+1):
            for  m in range(-premik, premik+1):
                for  l in range(-premik, premik+1):
                    t=np.exp(-2*np.pi*1j*m*k/float(N))*fat_delta(N,2*l-2*n+m)*psi[cycle(N,l+m)]*np.conjugate(psi[cycle(N,l)])/float(N)
                    w[n+premik][k+premik]+=t
    w=w*np.sqrt(N/(N-1)) 
    return w

@jit(nopython=pospesi) #wigner function for N^2xN^2 state, which is coupled
def get_wigner2(N,psi):
    w=np.zeros((N*N,N*N),dtype=np.complex_)
    premik=int((N-1)/2)
    for n in range(-premik, premik+1):
        for k in range(-premik, premik+1):
            for nn in range(-premik, premik+1):
                for kk in range(-premik, premik+1):
                    for n_ in range(-premik, premik+1):
                        for l_ in range(-premik, premik+1):
                            for nn_ in range(-premik, premik+1):
                                for ll_ in range(-premik, premik+1):
                                    x2=(l_+premik)*N+ll_+premik
                                    x1=cycle(N,l_+n_)*N+cycle(N, ll_+nn_)
                                    t=np.exp(-2*np.pi*1j*n_*k/float(N))*np.exp(-2*np.pi*1j*nn_*kk/float(N))
                                    t=t*fat_delta(N, 2*l_-2*n+n_)*fat_delta(N, 2*ll_-2*nn+nn_)
                                    t=t*psi[x1]*np.conjugate(psi[x2])
                                    kje1=(n+premik)*N+k+premik
                                    kje2=(nn+premik)*N+kk+premik
                                    w[kje1][kje2]+=t/float(N**2)
    #w=w/np.sum(w)
    return w

@jit(nopython=pospesi) 
def small_wigner(N,W):#get NxN wigner from N^2xN^2wigner
    w=np.zeros((N,N),dtype=np.complex_)
    for i in range(0,N):
        for j in range(0,N):
            vsota=0+0j
            for  k in range(0,N*N):
                vsota+=W[i*N+j][k]
            w[i][j]=vsota
    print(np.sum(w))
    return w/np.sum(w)

@jit(nopython=pospesi)  #dynamics
def eps(K,x):
    return -K/(2*np.pi)*np.sin(2*np.pi*(x))

@jit(nopython=pospesi) 
def kappa(Kc,q1,q2):   #dynamics
    return -Kc/(2*np.pi)*np.sin(2*np.pi*(q1)+2*np.pi*(q2))

@jit(nopython=pospesi) #dynamics
def C(N,Kc, j1,j2):
    return np.exp((1j*float(N)*Kc/(2*np.pi))*np.cos(2*np.pi/float(N)*(j1+j2)))

@jit(nopython=pospesi) #dynamics 
def A (N,M):
    return np.sqrt(1/(1j*float(N)*M[0][1]))

@jit(nopython=pospesi) #dynamics
def F(N,K,j):
    return (1j*K*float(N)/(2*np.pi))*np.cos(2*np.pi*j/float(N))

@jit(nopython=pospesi) 
def U(N,M,K,j,k):     #dynamics
    return A(N,M)*np.exp(1j*np.pi/(float(N)*M[0][1])*(M[0][0]*j**2-2*j*k+M[1][1]*k**2)+F(N,K,j))

@jit(nopython=pospesi) #join two quantum NxN state into a N^2xN^2 state    
def joint_psi(psi1,psi2):
    return np.kron(psi1,psi2)

@jit(nopython=pospesi) 
def coupled_cat_q(N,M,Kc,K,psi): #quantum propagator for N^2xN^2 state 
    prop=np.zeros((N*N,N*N),dtype=np.complex_)
    premik=int((N-1)/2)
    for j1 in range(-premik,premik+1):
        for j2 in range(-premik,premik+1):
            for k1 in range(-premik,premik+1):
                for k2 in range(-premik,premik+1):
                    x=(j1+premik)*N+j2+premik
                    y=(k1+premik)*N+k2+premik
                    prop[x][y]=U(N,M,K,j1,k1)*U(N,M,K,j2,k2)*C(N,Kc,j1,j2)
    psi=np.dot(prop, psi)
    return psi

@jit(nopython=pospesi) 
def entropy_q (N, psi):  #von Neumann entropy
    PSI=np.zeros((N,N),dtype=np.complex_)
    for i in range(0,N):
        for j in range(0,N):
            PSI[i][j]=psi[i*N+j]
    u, s, vh = np.linalg.svd(PSI, full_matrices=True)
    return get_e(s)

def entropy_qhat(N, psi):#rho operator entropy
    PSI=np.zeros([N*N,N*N])+0j
    for a1 in range(0,N):
        for a2 in range(0,N):
            for b1 in range(0,N):
                for b2 in range(0,N):
                    PSI[a1*N+a2][b1*N+b2]=psi[a1*N+b1]*np.conjugate(psi[a2*N+b2])
    u, s, vh = np.linalg.svd(PSI, full_matrices=True)
    return get_e(s)
    


@jit(nopython=pospesi)     
def c_density(N, Q,P,qsigma,psigma): #get gauss state for classical system
    ro=np.zeros((N,N))
    norm=0
    premik=int((N-1)/2)
    for i in range(Q-premik, Q+premik+1):
        for j in range(P-premik,P+premik+1):
            x=cycle(N,i)-premik
            y=cycle(N,j)-premik
            ro[x+premik][y+premik]=gauss(float(i)/float(N),Q/float(N),qsigma)*gauss(float(j)/float(N),P/float(N),psigma)
            norm+=ro[x+premik][y+premik]*ro[x+premik][y+premik]
    ro=ro/np.sqrt(norm)
    #ro=ro/np.sum(ro)
    return ro

@jit(nopython=pospesi) 
def joint_ro(N,ro1,ro2): #join two NxN classical state to N^2xN^2 state
    joined=np.zeros((N*N,N*N))
    for q1 in range(0,N):
        for p1 in range(0,N):
            for q2 in range(0,N):
                for p2 in range(0,N):
                    joined[q1+N*p1][q2+N*p2]+=ro1[q1][p1]*ro2[q2][p2]
    #joined=normiraj(joined)
    joined=joined/np.sum(joined)
    return joined

@jit(nopython=pospesi) 
def small_ro(N,ro): #get NxN classical density from N^2xN^2 state
    w=np.zeros((N,N),dtype=np.float_)
    for i in range(0,N):
        for j in range(0,N):
            vsota=0+0j
            for  k in range(0,N*N):
                vsota+=ro[i*N+j][k]
            w[i][j]=np.real(vsota)
    return w

@jit(nopython=pospesi) 
def entropy_c(N,ro): #get classical entropy, version 1
    u, s, vh = np.linalg.svd(ro, full_matrices=True)
    return get_e(s) #the difference between versions

@jit(nopython=pospesi) 
def entropy_c2(N,ro): #get classical entropy, version 1
    u, s, vh = np.linalg.svd(ro, full_matrices=True)
    return get_e2(s,ro)

@jit(nopython=pospesi) 
def coupled_cat_c(N,ro,M,Kc,K): #classical propagator for N^2xN^2 state
    premik=int((N-1)/2)
    new=np.zeros((N*N,N*N),dtype=np.float_)
    for Q1 in range(0,N):
        for P1 in range(0,N):
            for Q2 in range(0,N):
                for P2 in range(0,N):
                    q1=float(Q1-premik)/float(N)
                    q2=float(Q2-premik)/float(N)
                    p1=float(P1-premik)/float(N)
                    p2=float(P2-premik)/float(N)
                    prej1=np.array([q1, p1 + eps(K,q1)+kappa(Kc, q1,q2)])
                    prej2=np.array([q2, p2 + eps(K,q2)+kappa(Kc, q1,q2)])
                    koo1=np.dot(M,prej1)
                    koo2=np.dot(M,prej2)
                    Q_1,P_1=priredi(N,koo1)
                    Q_2,P_2=priredi(N,koo2)
                    new[Q_1*N+P_1][Q_2*N+P_2]+=ro[Q1*N+P1][Q2*N+P2]
    return new

@jit(nopython=pospesi)  
def priredi(N,koo):  #round the classical cooridantes to the closest discrete ones
    x=koo[0]
    y=koo[1]
    #print(x)
    x=float(N)*x
    y=float(N)*y
    x=cycle(N,int(round(x,0)))
    y=cycle(N,int(round(y,0)))
    return int(x),int(y)
    
@jit(nopython=pospesi) 
def get_e(s): #get the entropy from the diagonal elements from SVD
    vsota=0
    norma=0
    seznam=[]
    for i in range(0,len(s)):
        if(s[i]>=0):
            seznam.append(s[i])
            norma+=s[i]**2
    seznam=np.array(seznam)/np.sqrt(norma)
    for i in range(0,len(seznam)):
        x=seznam[i]**2
        vsota+=- x*np.log(x)
    return vsota

@jit(nopython=pospesi) 
def get_e2(s,ro): #get the entropy from the diagonal elements from SVD, but with different norm
    vsota=0
    norma=0
    seznam=[]
    for i in range(0,len(s)):
        if(s[i]>0):
            seznam.append(s[i])
            norma+=s[i]**2
    seznam=np.array(seznam)
    seznam=seznam/np.sqrt(norm(seznam))
    #seznam=seznam/np.sqrt(np.sum(RO**2))
    for i in range(0,len(seznam)):
        x=seznam[i]**2
        vsota+=- x*np.log(x)
    return vsota
   
@jit(nopython=pospesi)
def cutoff(N,ro):  #deleting the non-important elements from matrix
    RO=np.copy(ro)
    for i in range(0,N*N):
        for j in range(0,N*N):
            if(np.abs(ro[i][j])<10**(-15)):
                RO[i][j]=0
    return RO

N=21 #odd; 260s per time step for N=63, so 2600s for 10 time steps
Kc=0.5 #just like in Bergamasco
K=0.25 #just like in Bergamasco
M=Mh #choosing hyperbolic regimefro both systems

Q=0 #starting position in int
P=0 #starting position in int
q0=float(Q)/float(N) #starting position in float
p0=float(P)/float(N) #starting position in float


psi=get_psi_gauss(N,q0,p0,Q,P)  #get coherent state
h=1/(2*np.pi*float(N))          #get hbar
ro=c_density(N,Q,P,h**0.5,h**0.5) #get gauss state
ro=ro*ro #get the normalization right
PSI=joint_psi(psi,psi) #get quantum joint state
RO=joint_ro(N,ro,ro)   #get classical joint state


E1,E2,E3,E4,E5=[],[],[],[],[] #memorizing the results
for i in range(0,10): #number of time steps
    print('cas:'+str(time.time()-tau))
    E1.append(entropy_c2(N,cutoff(N,RO))) #classical entropy
    E2.append(entropy_q(N,PSI))           # quantum von neumann entropy
    RO=coupled_cat_c(N,RO,M,Kc,K)         #propagating classical state
    PSI=coupled_cat_q(N,M,Kc,K,PSI)        #propagating quantum state
E1.append(entropy_c2(N,cutoff(N,RO)))
E2.append(entropy_q(N,PSI))

E1=np.array(E1)
E2=np.array(E2)


#hello !!!



#plotting the curves, it should resemble Fig. 1 in Bergamasco 
import matplotlib.pyplot as plt
from matplotlib import  cm

plt.plot(E1/2)
plt.plot(E2)

plt.legend(['C','Q'])
print('cas:'+str(time.time()-tau))





'''
plt.plot(np.conj(psi)*psi)
plt.plot(np.conj((ft(N,psi)))*(ft(N,psi)))
plt.show()
plt.clf()

W=get_wigner(N,psi)
W=np.abs(W)
W=W/np.sum(W)
fig, ax = plt.subplots()
cs = ax.contourf(W, cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
ax.axis('equal')
plt.show()
print('cas:'+str(time.time()-tau))
plt.clf()

h=1/(2*np.pi*float(N))
RO=c_density(N,Q,P,h**0.5,h**0.5)
RO=RO*RO
fig, ax = plt.subplots()
cs = ax.contourf(RO, cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
ax.axis('equal')
plt.show()
'''
