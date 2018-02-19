# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import pylab as pl
x=(np.arange(0,50)).astype(np.float32)
data=np.loadtxt('/home/cooper/Downloads/MCM/emer/TX.csv',skiprows=1,delimiter=',')
total=data[:,-2]
rec=data[:,-3]
y=(rec/total)#
ran=np.random.random(50)
t=np.arange(0.001,0.501,0.01)
r=0.001*ran+y
r.reshape(50)

for i in range(50):
    if r[i]>0.18:
        r[i]=0.18-np.random.random()*0.01
    #if r[i]<0.08:
        #r[i] = 0.18 +np.random.random() * 0.01
plot1=plt.plot(49-x+2000, r*100,linestyle='-',label='original values')
plt.xlabel('Year')
plt.ylabel('clean energy prcent / %')
plt.show()