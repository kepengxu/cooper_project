# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
year=np.arange(50)+1960
dataMN=np.loadtxt('/home/cooper/Downloads/MCM/emer/MN.csv',skiprows=2,delimiter=',')
#RETCB=dataMN[:,23:24].reshape(50)
#CLTCB=dataMN[:,3:4].reshape(50)
#NGTCB=dataMN[:,17:18].reshape(50)
#PATCB=dataMN[:,19:20].reshape(50)
#print PATCB
RETCB=dataMN[:,24:25].reshape(50)
CLTCB=dataMN[:,3:4].reshape(50)
NGTCB=dataMN[:,17:18].reshape(50)
PATCB=dataMN[:,19:20].reshape(50)
print PATCB

width=0.8
p1=plt.bar(year,RETCB,width,color='yellow',bottom=0,label='RETCB')
p2=plt.bar(year,NGTCB,width,color='blue',bottom=RETCB,label='NGTCB')
p3=plt.bar(year,PATCB,width,color='green',bottom=RETCB+NGTCB,label='PATCB')
p4=plt.bar(year,CLTCB,width,color='red',bottom=RETCB+NGTCB+PATCB,label='CLTCB')
plt.legend((p1, p2,p3,p4), ('RETCB','NGTCB','PATCB','CLTCB'))
plt.xlabel('Year')
plt.ylabel('Energy consumpt/billion Btu')
plt.yticks(np.arange(0,900000,100000))
plt.show()



#width=0.8
#p1=plt.bar(year,RETCB,width,color='blue',label='RETCB')
#p2=plt.bar(year,CLTCB,width,color='green',bottom=RETCB,label='CLTCB')
#p3=plt.bar(year,NGTCB,width,color='yellow',bottom=CLTCB,label='NGTCB')
#p4=plt.bar(year,PATCB,width,color='red',bottom=NGTCB,label='PATCB')
#plt.xlabel('Year')
#plt.ylabel('Energy consumpt/billion Btu')
#plt.legend((p1, p2,p3,p4), ('RETCB','CLTCB','NGTCB','PATCB'))
#plt.yticks(np.arange(0,9000000,1000000))

plt.show()