from numba import jit
import numpy as np
import time
 
#measuring time
tau = time.time() 

#matrices for elliptic and hyperbolic behaviour
Mh=np.array([[2.0,1.0],[3.0,2.0]]) #hyperbolic
Me=np.array([[0.0,1],[-1,0.0]])    #elliptic

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

@jit(nopython=pospesi)  #JustInTime activation 
def norm (psi):  #returns norm of psi
    return np.sum(psi*np.conj(psi))

@jit(nopython=pospesi)  
def normiraj(psi):  #returns psi normed
    return psi/np.sqrt(norm(psi))
    
@jit(nopython=pospesi)     
def gauss(x,avg,sigma): #normal gauss function
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-avg)**2/(2*sigma**2))
  
@jit(nopython=pospesi)  
def get_psi_gauss(N,Q,P): #gauss state for quantum system
    q=np.zeros(N,dtype=np.complex_)
    q0=float(Q)/float(N)
    p0=float(P)/float(N)
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

@jit(nopython=pospesi) 
def entropy_qhat(N, psi):#rho operator entropy
    PSI=np.zeros((N*N,N*N),dtype=np.complex_)
    for a1 in range(0,N):
        for a2 in range(0,N):
            for b1 in range(0,N):
                for b2 in range(0,N):
                    PSI[a1*N+a2][b1*N+b2]=psi[a1*N+b1]*np.conjugate(psi[a2*N+b2])
    u, s, vh = np.linalg.svd(PSI, full_matrices=True)
    return get_e(s)
    


@jit(nopython=pospesi)     
def c_density(N, Q,P,qsigma,psigma): #get gauss state for the classical system
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
    return ro



@jit(nopython=pospesi) 
def joint_ro(N,ro1,ro2): #join two NxN classical state to N^2xN^2 state
    joined=np.zeros((N*N,N*N),dtype=np.float_)
    for q1 in range(0,N):
        for p1 in range(0,N):
            for q2 in range(0,N):
                for p2 in range(0,N):
                    joined[q1*N+p1][q2*N+p2]+=ro1[q1][p1]*ro2[q2][p2] 
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
def entropy_c(N,ro): #get classical entropy
    u, s, vh = np.linalg.svd(ro, full_matrices=True)
    return get_e(s) #calucate the entropy from acutal singular values


@jit(nopython=pospesi)  
def priredi(N,koo):
    x=koo[0]%1.0
    x=int(x*N)
    y=koo[1]%1.0
    y=int(y*N)
    return int(x),int(y)

@jit(nopython=pospesi) 
def coupled_cat_c_mat(N,M,Kc,K): #classical propagator for N^2xN^2 state; here we (pre)calculate the matrices Ar1 and Ar2
    #premik=int((N-1)/2) #in original article, middle is (0.5, 0.5). Here we decide, that our middle is actually (0, 0)
    premik=0    #here we decide, that our middle is (0.5, 0.5)
    new1=np.zeros((N*N,N*N),dtype=np.float_)
    new2=np.zeros((N*N,N*N),dtype=np.float_)
    for Q1 in range(0,N):
        for P1 in range(0,N):
            for Q2 in range(0,N):
                for P2 in range(0,N):
                    q1=float(Q1-premik)/float(N)
                    q2=float(Q2-premik)/float(N)
                    p1=float(P1-premik)/float(N)
                    p2=float(P2-premik)/float(N)
                    coo1=np.array([q1, p1 + eps(K,q1)+kappa(Kc, q1,q2)]) #first calculating the dynamics..
                    coo2=np.array([q2, p2 + eps(K,q2)+kappa(Kc, q1,q2)])
                    coo1=np.dot(M,coo1) #...multiplication by M
                    coo2=np.dot(M,coo2)
                    Q_1,P_1=priredi(N,coo1) #...getting coordinates back on torus
                    Q_2,P_2=priredi(N,coo2)
                    new1[Q1*N+P1][Q2*N+P2]=Q_1*N+P_1 #memorizing the new coordinates
                    new2[Q1*N+P1][Q2*N+P2]=Q_2*N+P_2
    return new1,new2

def coupled_cat_c_prop(N,ro,Ar1,Ar2): #propagating classical state via precalculated matrices Ar1 and Ar2 
    new=np.zeros((N*N,N*N),dtype=np.float_)
    for Q1 in range(0,N):
        for P1 in range(0,N):
            for Q2 in range(0,N):
                for P2 in range(0,N):
                    koo1=int(Ar1[Q1*N+P1][Q2*N+P2]) #new Q coordinate for the probability on [Q1*N+P1][Q2*N+P2]
                    koo2=int(Ar2[Q1*N+P1][Q2*N+P2]) #new P coordinate for the probability on [Q1*N+P1][Q2*N+P2]
                    new[koo1][koo2]+=ro[Q1*N+P1][Q2*N+P2] #we just move and add the probabilities together
    return new #new matrix of probabilities
    
@jit(nopython=pospesi) 
def get_e(s): #get the entropy from the diagonal elements from SVD
    entropy=0
    norma=0
    seznam=[]
    for i in range(0,len(s)): 
        if(s[i]>=0): #choosing only positive ones
            seznam.append(s[i])
            norma+=s[i]**2
    seznam=np.array(seznam)/np.sqrt(norma)
    for i in range(0,len(seznam)):
        x=seznam[i]**2
        entropy+=- x*np.log(x) 
    return entropy


N=11 #odd; 260s per time step for N=63, so 2600s for 10 time steps
Kc=0.5 #just like in Bergamasco
K=0.25 #just like in Bergamasco
M=Mh #choosing hyperbolic regime for both systems
#M=Me #choosing elliptic regime for both systems

Q=int(0) #starting position in int, 0 means middle
P=int(0) #starting position in int, 0 means middle


psi=get_psi_gauss(N,Q,P)  #get coherent N state (position representation)
h=1/(2*np.pi*float(N))    #get hbar
ro=c_density(N,Q,P,h**0.5,h**0.5) #get gauss NxN state
ro=ro*ro #get the normalization right
PSI=joint_psi(psi,psi) #get quantum joint  NxX state (position representation)
RO=joint_ro(N,ro,ro)   #get classical joint N²xN² state

Ar1,Ar2=coupled_cat_c_mat(N,M,Kc,K) #precalculate the matrices for propagating classical state. They stay the same throught the propagation.

E1,E2=[],[] #memorizing the results
for i in range(0,10): #number of time steps
    E1.append(entropy_c(N,RO**0.5)) #classical entropy (sqrt is used because of normalization)
    E2.append(entropy_q(N,PSI))     #quantum entropy
    PSI=coupled_cat_q(N,M,Kc,K,PSI)         #propagating quantum state
    RO=coupled_cat_c_prop(N,RO,Ar1,Ar2) #propagating classical state
E1.append(entropy_c(N,RO**0.5))
E2.append(entropy_q(N,PSI))

E1=np.array(E1)/2 #von neumann entropy is only a half of other entropies
E2=np.array(E2)


#plotting the curves, it should resemble Fig. 1 in Bergamasco 
import matplotlib.pyplot as plt
from matplotlib import  cm
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

print('cas:'+str(time.time()-tau))
plt.plot(E1) #plotting the curves
plt.plot(E2)
plt.ylabel('Entropy')
plt.xlabel('Time')
plt.legend(['classical','quantum'])


'''
To get pictures of phase space, just use the code below. Warning! Calculating
Winger functions for N²xN² system is VERY slow, max N=11 is recommended. Sorry!

get wigner for NxN system: w=get_wigner(N,psi)
get wigner for N²xN² system: W=get_wigner2(N,PSI)
get NxN wigner from N²xN² system: w=small_wigner(N,W) 

get NxN rho from N²xN² rho: ro=small_ro(N,RO)
  

Too see the results, just use this:
    
state=W # instead of W you can write w, ro, RO.
fig, ax = plt.subplots()
cs = ax.contourf(state, cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('Probability')
ax.axis('equal')
plt.ylabel('Position')
plt.xlabel('Momentum')
plt.tight_layout()
'''
