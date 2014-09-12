__author__ = 'yutongpang'
import numpy as np
class SVD():
    def __init__(self,n):
        self.n=n
    def kernel(self,s,t):  #expression of the kernel for the first type Fredholm integral equation of the first kind
        d=0.25
        value = d*(d**2+(s-t)**2)**(-3/2)
        return value
    def am(self):  # calculate the matrix expression for the operater
        n=self.n
        j = np.arange(1,n+1,dtype=np.float)
        sj = (j-1/2)/n
        tj = (j-1/2)/n
        s = (n,n)
        AM = np.zeros(s)
        for i in range(0,n):
            for j in range(0,n):
                AM[i,j] = self.kernel(sj[i],tj[j])/n
        return AM



#numpy.linalg.svd
svdmatrix=SVD(64)
AM=svdmatrix.am()
U, s, V =np.linalg.svd(AM,full_matrices=True)

# exact b function
j = np.arange(1,64+1,dtype=np.float)
sj = (j-1/2)/64
b = -60*(sj-0.5)**2+30
ut =U.T
utb =np.zeros((64,))
for i in range(0,64):
    utb[i] = abs(np.dot(ut[i],b))
print(ut)

import pylab
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yscale('log')
x = range(1,65)
plt.scatter(x,utb)
plt.show()

