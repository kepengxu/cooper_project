import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('/home/cooper/Downloads/MCM/data_GDP.csv',delimiter=',')
print data[0,0]
print data.shape
x=data[0]
y=data[1]
print x.shape
print y.shape
plt.plot(x,y)
