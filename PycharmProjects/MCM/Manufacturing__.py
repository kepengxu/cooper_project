###History of manufacturing GDP(Four State)
import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('/home/cooper/Downloads/MCM/data_GDP.csv',delimiter=',')
print data[0,0]
print data.shape
x=data[0]
y1=data[5]
y2=data[6]
y3=data[7]
y4=data[8]
plt.plot(x,y1,color="blue",linewidth=2,linestyle="-",label='Arizona')
plt.plot(x,y2,color="red",linewidth=2,linestyle="-",label='California')
plt.plot(x,y3,color="green",linewidth=2,linestyle="-",label='Texas')
plt.plot(x,y4,color="yellow",linewidth=2,linestyle="-",label='New Mexico')
plt.xlabel('Year')
plt.ylabel('State manufacturing GDP/ Millions of current dollars')
plt.legend(loc='upper left')
#x,y2,'bs',x,y3,'g^',x,y4,'mp')

plt.title('History of manufacturing GDP(Four State)')

plt.show()
