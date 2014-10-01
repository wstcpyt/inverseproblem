__author__ = 'yutongpang'
from SVD import SVD
import numpy as np
n=17
lambdaarray = ['pdf/300.txt','pdf/350.txt','pdf/400.txt','pdf/450.txt','pdf/500.txt','pdf/550.txt','pdf/600.txt',
              'pdf/650.txt','pdf/700.txt','pdf/750.txt','pdf/800.txt','pdf/850.txt','pdf/900.txt','pdf/950.txt',
              'pdf/1000.txt','pdf/1050.txt','pdf/1100.txt','pdf/1150.txt']

xarray=['-0.000000000000000000e+00','6.264896152536601759e-02','1.249574395641811358e-01','1.876064010895471812e-01','2.505958461014640704e-01','3.125638406537283309e-01',
          '3.758937691521960778e-01','4.375212802179094251e-01','5.005107252298264253e-01','5.624787197820905194e-01','6.251276813074565508e-01',
          '6.874361593462717801e-01','7.507660878447396380e-01','8.127340823970038430e-01','8.750425604358188503e-01','9.376915219611848817e-01',
          '9.996595165134490868e-01']
wavelength = np.arange(300,1150,50)

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
f_tikhonov =svdclass.f_tikhonov(0.28,AM,g.T)
lcurvex,lcurvey=svdclass.lcurve(AM,g.T)
ginverse_tsvd = np.dot(AM,f_tsvd)
ginverse_tikhonov = np.dot(AM,f_tikhonov)
x = np.arange(0,n)
utb, utbs=svdclass.picardparameter(AM,g.T)
U, s, V = svdclass.svdmatrix(AM)


########collection coefficient from modeling########
iqescr=0.9
zmo=2300.0
leff=920.0
smooverd=4.3e-4
def iqefunction(z):
    numerator= iqescr*((np.cosh((z-zmo)/leff))/leff-smooverd*np.sinh((z-zmo)/leff))
    denominator= np.cosh((zmo-350.0)/leff)/leff+smooverd*np.sinh((zmo-350.0)/leff)
    iqevalue=numerator/denominator
    return iqevalue
iqe=np.zeros((2301,))
for i in range(0,351):
    iqe[i] = iqescr
for i in range(350,2301):
    iqe[i] = iqefunction(i)
iqex = np.arange(0,2301)



from pylab import *
ax1 = subplot(111)
#ax1.scatter(xarray,f_tikhonov,marker='o',label='tkhonov',color='black')
#ax1.scatter(xarray,f_tsvd,marker='+',label='tsvd',color='red')
ax1.scatter(iqex,iqe,marker='+',label='tsvd',color='red')
#ax1.scatter(wavelength,gm,marker='*',label='measured',color='green')
#ax1.set_xlim([-0.1,1.1])
ax1.set_xlabel('Wavelength(nm)')
ax1.set_ylabel('Internal Quantum Efficiency')
#ax1.set_title('Collection Coefficient(Not Normalized)')
legend = ax1.legend(loc='upper left', shadow=True,prop={'size':12})

show()
