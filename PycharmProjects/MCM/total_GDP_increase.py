##History growth rate of total GDP(Four State)
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
growth_rate=np.zeros([5,49],float)
for i in range(1,50):
    growth_rate[1, i-1] = (float(data[1, i]) - float(data[1, i - 1])) / float(data[1, i-1])
    print float(growth_rate[1, i-1])
    growth_rate[2, i-1] = (data[2, i] - data[2, i - 1]) / data[2, i-1]
    growth_rate[3, i-1] = (data[3, i] - data[3, i - 1]) / data[3, i-1]
    growth_rate[4, i-1] = (data[4, i] - data[4, i - 1]) / data[4, i-1]
plt.plot(x[0:49],growth_rate[1],color="blue",linewidth=1,linestyle="-",label='Arizona')
plt.plot(x[0:49],growth_rate[2],color="red",linewidth=1,linestyle="-",label='California')
plt.plot(x[0:49],growth_rate[3],color="orange",linewidth=1,linestyle="-",label='Texas')
plt.plot(x[0:49],growth_rate[4],color="yellow",linewidth=1,linestyle="-",label='New Mexico')
plt.xlabel('Year')
plt.ylabel('the growth rate of State total GDP/ Millions of current dollars')
plt.legend(loc='upper left')
#x,y2,'bs',x,y3,'g^',x,y4,'mp')

plt.title('History growth rate of total GDP(Four State)')
print growth_rate[1,1]
plt.show()
print (data[1,1]-data[1,0])/data[1,0]