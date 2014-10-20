__author__ = 'yutongpang'
import numpy as np
import matplotlib.cm as cm
from pylab import *
ax1 = subplot(111)
wl = np.array([300,500,700,900,1100])
i=0
for wlobject in wl:
    pdfx=np.array([])
    pdfy=np.array([])
    iqetxtfile = open('pdf/%d.txt'%(wlobject))
    for line in iqetxtfile:
        pdfx = np.append(pdfx,[float(line.split()[0])])
        pdfy = np.append(pdfy,[float(line.split()[1])])
    ax1.plot(pdfx,pdfy,linewidth=3.0,label='wl=%dnm'%(wlobject))
    i = i+1
legend = ax1.legend(loc='upper right', shadow=True,prop={'size':15})
ax1.set_ylim(0,22)
ax1.set_xlabel('normalized position in device',fontsize=15)
ax1.set_ylabel('normalized generation function',fontsize=15)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)

show()
