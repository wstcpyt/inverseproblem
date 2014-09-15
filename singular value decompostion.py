__author__ = 'yutongpang'
import numpy as np
class SVD():
    def __init__(self,n):
        self.n=n
    def kernel(self,s,t):  #expression of the kernel for the first type Fredholm integral equation of the first kind
        d=0.5
        value = d*(d**2+(s-t)**2)**(-3/2)
        return value
    def am(self):  # calculate the matrix expression for the operater
        n=self.n
        j = np.arange(0,n,dtype=np.float)
        sj = (j)/n
        tj = (j)/n
        s = (n,n)
        AM = np.zeros(s)
        for i in range(0,n):
            for j in range(0,n):
                AM[i,j] = self.kernel(sj[i],tj[j])/n
        return AM
    def g(self,AM):
        n= self.n
        f = np.zeros((n,))
        for i in range(0,30):
            f[i] = 2
        for i in range (30,n):
            f[i] = 1
        gvalue = np.dot(AM,f)
        return gvalue



#numpy.linalg.svd
svdmatrix=SVD(64)
AM=svdmatrix.am()
g =svdmatrix.g(AM)
U, s, V =np.linalg.svd(AM,full_matrices=True)


ut =U.T
utb =np.zeros((64,))
for i in range(0,64):
    utb[i] = (np.dot(ut[i],g))/s[i]
print(V)
xfilter = 0
for i in range(0,20):
    xfilter = xfilter + utb[i]*V[i]
print(xfilter)
import pylab
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.set_yscale('log')
x = range(1,65)
plt.scatter(x,xfilter)
plt.show()

