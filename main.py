from numba import jit
import numpy as np
import time
 
#measuring time
tau = time.time() 

#matrices for elliptic and hyperbolic behaviour
Mh=np.array([[2.0,1],[3,2]])
Me=np.array([[0.0,1],[-1,0.0]])

pospesi=True # @jit(nopython=pospesi)  <- activating JustInTime compilation for faster computing
    
def ft(N,psi):
    psi2=np.zeros([N])+0j
    for i in range(0,N):
        for j in range(0,N):
            psi2[i]+=1/np.sqrt(N)*np.exp(-2*np.pi*1j*j*i/N)*psi[j]
    return psi2

def ift(N,psi):
    psi2=np.zeros([N])+0j
    for i in range(0,N):
        for j in range(0,N):
            psi2[i]+=1/np.sqrt(N)*np.exp(2*np.pi*1j*j*i/N)*psi[j]
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
    sigma=1.0
    for i in range(0,N):
        for j in range(0,N):
            q[i]+=np.exp(1j*2*np.pi*p0*float(i))*np.exp(-(np.pi*sigma/float(N))*(float(i)-q0*float(N)+float(j)*N)**2)*np.exp(1j*2*np.pi*p0*float(j)*float(N))
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
    while a<0:
        a+=N
    while a>=N:
        a-=N
    return a
    

@jit(nopython=pospesi)  #dynamics
def eps(K,x):
    return -K/(2*np.pi)*np.sin(2*np.pi*(float(x)))

@jit(nopython=pospesi) 
def kappa(Kc,q1,q2):   #dynamics
    return -Kc/(2*np.pi)*np.sin(2*np.pi*float(q1)+2*np.pi*float(q2))

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
    for j1 in range(0,N):
        for j2 in range(0,N):
            for k1 in range(0,N):
                for k2 in range(0,N):
                    x=j1*N+j2
                    y=k1*N+k2
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
def c_density(N, Q,P,qsigma,psigma): #get gauss state for classical system
    ro=np.zeros((N,N))
    norm=0
    half=int(N/2)
    for i in range(Q-half, Q+half):
        for j in range(P-half,P+half):
            x=cycle(N,i)
            y=cycle(N,j)
            ro[x][y]=gauss(float(i)/float(N),Q/float(N),qsigma)*gauss(float(j)/float(N),P/float(N),psigma)
            norm+=ro[x][y]*ro[x][y]
    ro=ro/np.sqrt(norm)
    return ro

@jit(nopython=pospesi) 
def joint_ro(N,ro1,ro2): #join two NxN classical state to N^2xN^2 state
    joined=np.zeros((N*N,N*N))
    for q1 in range(0,N):
        for p1 in range(0,N):
            for q2 in range(0,N):
                for p2 in range(0,N):
                    joined[q1*N+p1][q2*N+p2]+=ro1[q1][p1]*ro2[q2][p2]
    #joined=normiraj(joined)
    #joined=joined/np.sum(joined)
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
    new=np.zeros((N*N,N*N),dtype=np.float_)
    for Q1 in range(0,N):
        for P1 in range(0,N):
            for Q2 in range(0,N):
                for P2 in range(0,N):
                    q1=float(Q1)/float(N)
                    q2=float(Q2)/float(N)
                    p1=float(P1)/float(N)
                    p2=float(P2)/float(N)
                    prej1=np.array([q1, p1 + eps(K,q1)+kappa(Kc, q1,q2)])
                    prej2=np.array([q2, p2 + eps(K,q2)+kappa(Kc, q1,q2)])
                    koo1=np.dot(M,prej1)
                    koo2=np.dot(M,prej2)
                    Q_1,P_1=priredi(N,koo1)
                    Q_2,P_2=priredi(N,koo2)
                    new[Q_1*N+P_1][Q_2*N+P_2]=(ro[Q1*N+P1][Q2*N+P2]**2+new[Q_1*N+P_1][Q_2*N+P_2]**2)**0.5
                    
    return new

@jit(nopython=pospesi)  
def priredi(N,koo): #round the classical cooridantes to the closest discrete ones
    koo%=1.0
    return int(round(koo[0]*float(N),0)),int(round(koo[1]*float(N),0))
    
@jit(nopython=pospesi) 
def get_e(s): #get the entropy from the diagonal elements from SVD
    vsota=0
    norma=0
    seznam=[]
    for i in range(0,len(s)):
        if(s[i]>0):
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

def join_exp(N,psi1,psi2):
    w=np.zeros((N,N),dtype=np.float_)
    for i in range(0,N):
        for j in range(0,N):
            w[i][j]=psi1[i]*psi2[j]
    return w

N=2**4
Kc=0.5 #just like in Bergamasco
K=0.25 #just like in Bergamasco
M=Mh #choosing hyperbolic regimefro both systems

Q=int(N/2) #starting position in int
P=int(N/2)#starting position in int
q0=float(Q)/float(N) #starting position in float
p0=float(P)/float(N) #starting position in float


psi=get_psi_gauss(N,q0,p0,Q,P)  #get coherent state
h=1/(2*np.pi*float(N))          #get hbar
#ro=c_density(N,Q,P,h**0.5,h**0.5) #get gauss state
ro=c_density(N,Q,P,h**0.5,h**0.5)
#ro=ro*ro #get the normalization right
PSI=joint_psi(psi,psi) #get quantum joint state
RO=joint_ro(N,ro,ro)   #get classical joint state
#RO=join_exp(N*N, np.real(np.conj(PSI)*PSI), np.real(np.conj((ft(N*N,PSI)))*(ft(N*N,PSI))))
#ro=join_exp(N, np.real(np.conj(psi)*psi), np.real(np.conj((ft(N,psi)))*(ft(N,psi))))
#RO=np.sqrt(RO)


E1,E2,E3,E4,E5=[],[],[],[],[] #memorizing the results
for i in range(0,10): #number of time steps
    print('cas:'+str(time.time()-tau))
    E1.append(entropy_c(N,RO*RO)) #classical entropy
    E2.append(entropy_q(N,PSI))           # quantum von neumann entropy
    RO=coupled_cat_c(N,RO,M,Kc,K)         #propagating classical state
    PSI=coupled_cat_q(N,M,Kc,K,PSI)        #propagating quantum state
E1.append(entropy_c(N,RO*RO))
E2.append(entropy_q(N,PSI))

E1=np.array(E1)/2
E2=np.array(E2)


#hello !!!



#plotting the curves, it should resemble Fig. 1 in Bergamasco 
import matplotlib.pyplot as plt
from matplotlib import  cm

plt.plot(E1)
plt.plot(E2)

plt.legend(['C','Q'])
print('cas:'+str(time.time()-tau))




(0.8272200041845494, 13.235520066952791, 13)



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
