__author__ = 'yutongpang'
import numpy as np
pdfx=np.array([])
pdfy=np.array([])
iqetxtfile = open('pdf/1000.txt')
for line in iqetxtfile:
    pdfx = np.append(pdfx,[float(line.split()[0])])
    pdfy = np.append(pdfy,[float(line.split()[1])])
from pylab import *
ax1 = subplot(111)
ax1.plot(pdfx,pdfy,label='tkhonov',color='black')
show()
