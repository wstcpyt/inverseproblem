__author__ = 'yutongpang'
from SVD import SVD
import numpy as np
n=17
lambdaarray = ['pdf/300.txt','pdf/350.txt','pdf/400.txt','pdf/450.txt','pdf/500.txt','pdf/550.txt','pdf/600.txt',
              'pdf/650.txt','pdf/700.txt','pdf/750.txt','pdf/800.txt','pdf/850.txt','pdf/900.txt','pdf/950.txt',
              'pdf/1000.txt','pdf/1050.txt','pdf/1100.txt','pdf/1150.txt']

xarray=['9.159005788219272415e-02','1.310861423220973931e-01','1.767109295199182917e-01','2.223357167177392180e-01','2.683009874021110019e-01','3.142662580864827859e-01',
        '3.598910452843037122e-01','4.058563159686755517e-01','4.450119169220292936e-01','4.974463738508682065e-01','5.434116445352400460e-01','5.893769152196118855e-01',
        '6.346612189308818985e-01','6.806264896152537380e-01','7.265917602996254665e-01','7.722165474974463928e-01','8.181818181818182323e-01']
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
f_tikhonov =svdclass.f_tikhonov(0.3,AM,g.T)
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
iqenormx=np.arange(0,1,1.0/2301)
print(len(iqenormx))



from pylab import *
ax1 = subplot(111)
ax1.scatter(xarray,f_tikhonov,marker='o',label='tkhonov',color='black')
ax1.scatter(xarray,f_tsvd,marker='+',label='tsvd',color='red')
ax1.scatter(iqenormx,iqe,marker='+',label='best fit',color='green')
#ax1.scatter(wavelength,gm,marker='*',label='measured',color='green')
#ax1.set_xlim([-0.1,1.1])
ax1.set_xlabel('Wavelength(nm)')
ax1.set_ylabel('Internal Quantum Efficiency')
#ax1.set_title('Collection Coefficient(Not Normalized)')
#legend = ax1.legend(loc='upper left', shadow=True,prop={'size':12})

show()
