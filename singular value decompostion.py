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
    def g(self,AM):  # measured function
        n= self.n
        f = np.zeros((n,))
        for i in range(0,30):
            f[i] = 2
        for i in range (30,n):
            f[i] = 1
        gvalue = np.dot(AM,f)
        return gvalue,f

    def svdmatrix(self,AM):   #svd decomposition
        U, s, V = np.linalg.svd(AM,full_matrices=True)
        return U, s, V

    def picardparameter(self,AM,g): #picadrd parameter
        n = self.n
        utb = np.zeros((n,))
        utbs = np.zeros((n,))
        U, s, V = self.svdmatrix(AM)
        ut = U.T
        for i in range(0,n):
            utb[i] = (np.dot(ut[i],g))
            utbs[i] = (np.dot(ut[i],g))/s[i]

        return utb, utbs

    def ffilter(self,filtern,AM,g):  #reconstruction functtion
        ffilter = 0
        utb, utbs = self.picardparameter(AM,g)
        U,s,V=self.svdmatrix(AM)
        for i in range(0,filtern):
            ffilter = ffilter + utbs[i]*V[i]
        return ffilter


n=17
lambdaarray = ['data/300.txt','data/350.txt','data/400.txt','data/450.txt','data/500.txt','data/550.txt','data/600.txt',
             'data/650.txt','data/700.txt','data/750.txt','data/800.txt','data/850.txt','data/900.txt','data/950.txt',
             'data/1000.txt','data/1050.txt','data/1100.txt','data/1150.txt']
xarray=['-0.35','-0.477','-0.604','-0.731','-0.858','-0.985','-1.112','-1.239','-1.366','-1.493','-1.62','-1.747','-1.874',
        '-2.001','-2.128','-2.255','-2.382','-2.509']
g= np.matrix([1.344,11.511,52.300,67.217,88.613,93.901,94.127,94.191,93.928,93.227,92.437,91.066,88.771,84.937,72.927,41.473,9.945])
print(g)
s=(n,n)
AM = np.zeros(s)
for i in range(0,n):
    for j in range(0,n):
        search = open(lambdaarray[i])
        for line in search:
            if line.split()[0]==xarray[j]:
                AM[i,j] = float(line.split()[1])

svdclass=SVD(17)
f_tsvd = svdclass.ffilter(4,AM,g.T)
print(f_tsvd)
x = np.arange(0,n)
utb, utbs=svdclass.picardparameter(AM,g.T)
U, s, V = svdclass.svdmatrix(AM)

from pylab import *
from matplotlib import rc
ax = subplot(111)
#ax.set_yscale('log')
ax.scatter(x,f_tsvd,marker='o',label=r'$\left| u_{i}^{T}b \right|$',color='black')
#ax.scatter(x,abs(utbs),marker='x',label=r'$\left| u_{i}^{T}b \right|/{{\sigma }_{i}}$',color='red')
#ax.scatter(x,s,label=r'$\sigma_{i}}$',marker='+',color='green',s=60)
#ax.set_xlim([-1,20])
ax.set_xlabel('i')
ax.set_title('Collection Coefficient(Not Normalized)')
#legend = ax.legend(loc='upper left', shadow=True)
show()







