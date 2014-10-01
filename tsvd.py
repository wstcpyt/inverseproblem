__author__ = 'yutongpang'
from SVD import SVD
import numpy as np
n=17
lambdaarray = ['pdf/300.txt','pdf/350.txt','pdf/400.txt','pdf/450.txt','pdf/500.txt','pdf/550.txt','pdf/600.txt',
              'pdf/650.txt','pdf/700.txt','pdf/750.txt','pdf/800.txt','pdf/850.txt','pdf/900.txt','pdf/950.txt',
              'pdf/1000.txt','pdf/1050.txt','pdf/1100.txt','pdf/1150.txt']

# xarray=['-0.000000000000000000e+00','6.264896152536601759e-02','1.249574395641811358e-01','1.876064010895471812e-01','2.505958461014640704e-01','3.125638406537283309e-01',
#          '3.758937691521960778e-01','4.375212802179094251e-01','5.005107252298264253e-01','5.624787197820905194e-01','6.251276813074565508e-01',
#          '6.874361593462717801e-01','7.507660878447396380e-01','8.127340823970038430e-01','8.750425604358188503e-01','9.376915219611848817e-01',
#          '9.996595165134490868e-01']
xarray=['9.159005788219272415e-02','1.310861423220973931e-01','1.767109295199182917e-01','2.223357167177392180e-01','2.683009874021110019e-01','3.142662580864827859e-01',
        '3.598910452843037122e-01','4.058563159686755517e-01','4.450119169220292936e-01','4.974463738508682065e-01','5.434116445352400460e-01','5.893769152196118855e-01',
        '6.346612189308818985e-01','6.806264896152537380e-01','7.265917602996254665e-01','7.722165474974463928e-01','8.181818181818182323e-01']
g= np.array([1.344,11.511,52.300,67.217,88.613,93.901,94.127,94.191,93.928,93.227,92.437,91.066,88.771,84.937,72.927,41.473,9.945])

g=g/100
gm =np.array([1.344/100,11.511/100,52.300/100,67.217/100,88.613/100,93.901/100,94.127/100,94.191/100,93.928/100,93.227/100,92.437/100,91.066/100,88.771/100,84.937/100,72.927/100,41.473/100,9.945/100])
print(g)
s=(n,n)
AM = np.zeros(s)
for i in range(0,n):
    for j in range(0,n):
        search = open(lambdaarray[i])
        for line in search:
            if line.split()[0]==xarray[j]:
                AM[i,j] = float(line.split()[1])*1.0/n

svdclass=SVD(n)
f_tsvd = svdclass.f_tsvd(3,AM,g.T)
f_tikhonov =svdclass.f_tikhonov(0.5,AM,g.T)
lcurvex,lcurvey=svdclass.lcurve(AM,g.T)
ginverse_tsvd = np.dot(AM,f_tsvd)
ginverse_tikhonov = np.dot(AM,f_tikhonov)
x = np.arange(0,n)
utb, utbs=svdclass.picardparameter(AM,g.T)
U, s, V = svdclass.svdmatrix(AM)


from pylab import *
from matplotlib import rc
ax = subplot(111)
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.scatter(x,ginverse_tsvd,marker='o',label='tkhonov',color='black')
ax.scatter(x,f_tsvd,marker='o',label='measured',color='red')
#ax.scatter(x,gm,marker='o',label='measured',color='green')
#ax.scatter(x,abs(utbs),marker='x',label=r'$\left| u_{i}^{T}b \right|/{{\sigma }_{i}}$',color='red')
#ax.scatter(x,s,label=r'$\sigma_{i}}$',marker='+',color='green',s=60)
#ax.set_xlim([-1,20])
ax.set_xlabel('i')
ax.set_title('Collection Coefficient(Not Normalized)')
legend = ax.legend(loc='upper left', shadow=True)
show()
