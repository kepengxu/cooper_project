"""
about draw with scatter
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
N=100
x=np.random.randn(N)
y=np.random.randn(N)
color=['r','y','k','g','m']
plt.scatter(x,y,c='g',marker='x',alpha=0.5,edgecolors='g')
plt.title('random')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()