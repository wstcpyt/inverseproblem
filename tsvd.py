__author__ = 'yutongpang'
from SVD import SVD
import numpy as np
n=17
lambdaarray = ['pdf/300.txt','pdf/350.txt','pdf/400.txt','pdf/450.txt','pdf/500.txt','pdf/550.txt','pdf/600.txt',
             'pdf/650.txt','pdf/700.txt','pdf/750.txt','pdf/800.txt','pdf/850.txt','pdf/900.txt','pdf/950.txt',
             'pdf/1000.txt','pdf/1050.txt','pdf/1100.txt','pdf/1150.txt']
#xarray=['-0.35','-0.477','-0.604','-0.731','-0.858','-0.985','-1.112','-1.239','-1.366','-1.493','-1.62','-1.747','-1.874',
#        '-2.001','-2.128','-2.255','-2.382','-2.509']
xarray=['-0.000000000000000000e+00','6.264896152536601759e-02','1.249574395641811358e-01','1.876064010895471812e-01','2.505958461014640704e-01','3.125638406537283309e-01',
        '3.758937691521960778e-01','4.375212802179094251e-01','5.005107252298264253e-01','5.624787197820905194e-01','6.251276813074565508e-01',
        '6.874361593462717801e-01','7.507660878447396380e-01','8.127340823970038430e-01','8.750425604358188503e-01','9.376915219611848817e-01',
        '9.996595165134490868e-01']
g= np.matrix([1.344,11.511,52.300,67.217,88.613,93.901,94.127,94.191,93.928,93.227,92.437,91.066,88.771,84.937,72.927,41.473,9.945])
gm =[1.344,11.511,52.300,67.217,88.613,93.901,94.127,94.191,93.928,93.227,92.437,91.066,88.771,84.937,72.927,41.473,9.945]
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
f_tsvd = svdclass.f_tsvd(3,AM,g.T)
f_tikhonov =svdclass.f_tikhonov(1,AM,g.T)
lcurvex,lcurvey=svdclass.lcurve(AM,g.T)
ginverse_tsvd = np.dot(AM,f_tsvd)
ginverse_tikhonov = np.dot(AM,f_tikhonov)
x = np.arange(0,n)
utb, utbs=svdclass.picardparameter(AM,g.T)
U, s, V = svdclass.svdmatrix(AM)


from pylab import *
from matplotlib import rc
ax = subplot(111)
ax.set_yscale('log')
ax.set_xscale('log')
ax.scatter(lcurvex,lcurvey,marker='o',label='tkhonov',color='black')
#ax.scatter(x,gm,marker='o',label='measured',color='red')
#ax.scatter(x,gm,marker='o',label='measured',color='green')
#ax.scatter(x,abs(utbs),marker='x',label=r'$\left| u_{i}^{T}b \right|/{{\sigma }_{i}}$',color='red')
#ax.scatter(x,s,label=r'$\sigma_{i}}$',marker='+',color='green',s=60)
#ax.set_xlim([-1,20])
ax.set_xlabel('i')
ax.set_title('Collection Coefficient(Not Normalized)')
legend = ax.legend(loc='upper left', shadow=True)
show()