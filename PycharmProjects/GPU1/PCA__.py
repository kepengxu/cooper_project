import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
data=np.loadtxt('/home/cooper/Downloads/MCM/buquan_exp_CSV.csv',delimiter=',',skiprows=2)
#print data.shape
di=np.arange(90)+10
sum_loss_abs=np.arange(90)
for i in range(10,100):
    pca=PCA(n_components=i,whiten=True)
    newData=pca.fit_transform(data)
#    print newData.shape
#    print np.sum(pca.explained_variance_ratio_)
#    print pca.explained_variance_ratio_
    M=data
#    print "Mshape",M.shape
    t=pca.transform(M)
    inverse=pca.inverse_transform(t)
    emmm=np.sum(abs(M-inverse))
    sum_loss_abs[i-10]=emmm
    print i,emmm

plt.plot(di,sum_loss_abs,linewidth=2,color='yellow')
plt.title('Relationship Between Loss Value and Retention Dimension after Dimension Reduction')
plt.xlabel('Reserved dimensions')
plt.ylabel('Total loss value')
plt.show()