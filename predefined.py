__author__ = 'yutongpang'
########collection coefficient from modeling########
import numpy as np
iqescr=0.9
zmo=1
leff=920.0/2300
smooverd=4.3e-4
def iqefunction(z):
    numerator= iqescr*((np.cosh((z-zmo)/leff))/leff-smooverd*np.sinh((z-zmo)/leff))
    denominator= np.cosh((zmo-350.0/2300)/leff)/leff+smooverd*np.sinh((zmo-350.0/2300)/leff)
    iqevalue=numerator/denominator
    return iqevalue
xarray=['8.580183861082738006e-02','1.310861423220973931e-01','1.767109295199182917e-01','2.223357167177392180e-01','2.683009874021110019e-01','3.142662580864827859e-01',
        '3.598910452843037122e-01','4.058563159686755517e-01','4.450119169220292936e-01','4.974463738508682065e-01','5.434116445352400460e-01','5.893769152196118855e-01',
        '6.346612189308818985e-01','6.806264896152537380e-01','7.265917602996254665e-01','7.722165474974463928e-01','8.181818181818182323e-01']
cc = np.array([])
for xarrayobject in xarray:
    if float(xarrayobject)<350.0/2300:
        cc = np.append(cc,0.9)
    else:
        cc = np.append(cc,iqefunction(float(xarrayobject)))
print(cc)
n=17
lambdaarray = ['pdf/300.txt','pdf/350.txt','pdf/400.txt','pdf/450.txt','pdf/500.txt','pdf/550.txt','pdf/600.txt',
              'pdf/650.txt','pdf/700.txt','pdf/750.txt','pdf/800.txt','pdf/850.txt','pdf/900.txt','pdf/950.txt',
              'pdf/1000.txt','pdf/1050.txt','pdf/1100.txt','pdf/1150.txt']
s=(n,n)
AM = np.zeros(s)
for i in range(0,n):
    for j in range(0,n):
        search = open(lambdaarray[i])

        for line in search:
            if line.split()[0]==xarray[j]:
                AM[i,j] = float(line.split()[1])*1.0/n

iqe = np.dot(AM,cc)
print(iqe)


filename = open('pdf/300.txt')
totaliqe= 0
modnumber = 0

for line in filename:
    if modnumber%143==0:
        totaliqe = totaliqe + float(line.split()[1])
    modnumber = modnumber +1

print(totaliqe*143/2937)




from pylab import *
ax1 = subplot(111)
ax1.scatter(xarray,iqe,marker='o',label='tkhonov',color='black')


show()