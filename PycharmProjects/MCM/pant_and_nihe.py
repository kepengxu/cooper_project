##History of total GDP(Four State)
import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('/home/cooper/Downloads/MCM/data_GDP.csv',delimiter=',')
print data[0,0]
print data.shape
x=data[0]
y1=data[1]
y2=data[2]
y3=data[3]
y4=data[4]
plt.plot(x,y1,color="blue",linewidth=2,linestyle="-",label='Arizona')
plt.plot(x,y2,color="red",linewidth=2,linestyle="-",label='California')
plt.plot(x,y3,color="orange",linewidth=2,linestyle="-",label='Texas')
plt.plot(x,y4,color="yellow",linewidth=2,linestyle="-",label='New Mexico')
plt.xlabel('Year')
plt.ylabel('State total GDP/ Millions of current dollars')
plt.legend(loc='upper left')
#x,y2,'bs',x,y3,'g^',x,y4,'mp')

plt.title('History of total GDP(Four State)')

plt.show()
