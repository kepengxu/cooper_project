import numpy as np
import matplotlib.pyplot as plt
N=100
x=np.random.randn(N)
y=np.random.randn(N)
color=['r','y','k','g','m']
plt.scatter(x,y,c=color,marker='>')
plt.show()