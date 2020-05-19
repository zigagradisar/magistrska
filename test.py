from numba import jit
import numpy as np
import time


N=2**9
a=0.5
b=2
pospesi=True
meja=2*np.pi
h=meja/float(N)
@jit(nopython=pospesi) 
def get_ro(N,meja):
    ro=np.zeros((N,N),dtype=np.float_)
    for i in range(0,N):
        for j in range(0,N):
            x=meja*float(i)/float(N)
            y=meja*float(j)/float(N)
            ro[i][j]=(2+np.cos(x)+np.cos(y))*(2*np.pi*np.sqrt(5))
    #ro=ro/np.sqrt(np.sum(ro*ro))
    return ro

@jit(nopython=pospesi) 
def action(N,ro,meja,a,b):
    for t in range(0,3):
        new=np.zeros((N,N),dtype=np.float_)
        for i in range(0,N):
            for j in range(0,N):
                x=meja*float(i)/float(N)
                y=meja*float(j)/float(N)
                y_=y+x-a*np.sin(x)
                #y_=y+a*np.sin(x)+b*np.cos(2*x)
                y_%=meja
                x_=x+y_
                y_%=meja
                x_%=meja
                kjex=int(round((x_/meja)*N,0))%N
                kjey=int(round((y_/meja)*N,0))%N
                new[kjex][kjey]=(new[kjex][kjey]**2+ro[i][j]**2)**0.5
                #new[int((x_/meja)*N)][int((y_/meja)*N)]+=ro[i][j]
        ro=np.copy(new)
    return ro

ro=get_ro(N,meja)
ro=action(N,ro,meja,a,b)
ro=np.flipud(ro)
#ro=np.fliplr(ro)
#ro=np.roll(ro, -int(N/2), axis=1)

from scipy.ndimage.interpolation import rotate
ro = rotate(ro, angle=0)
import matplotlib.pyplot as plt
from matplotlib import  cm
fig, ax = plt.subplots()
cs = ax.contourf(ro, cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
ax.axis('equal')
plt.show()