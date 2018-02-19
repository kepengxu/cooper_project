import numpy as np
from sklearn.decomposition import PCA
data=np.loadtxt('/home/cooper/Downloads/MCM/buquan_exp_CSV.csv',delimiter=',',skiprows=2)
#print data.shape

pca=PCA(n_components=49,whiten=True)
newData=pca.fit_transform(data)
print newData.shape
print np.sum(pca.explained_variance_ratio_)
print pca.explained_variance_ratio_
M=data
print "Mshape",M.shape
t=pca.transform(M)
inverse=pca.inverse_transform(t)
time=0
print M.shape
for i in range(50):
    for j in range(432):
        cha=abs(M[i,j]-inverse[i,j])
        if cha>10000:
            time=time+1
            print i,j,cha
emmm=np.mean(abs(M-inverse))
print emmm
print time
#sum=0.0
#for i in range(432):
#    print abs(inverse[0,i]-M[0,i])
#    sum+=abs(inverse[0,i]-M[0,i])
#
#print sum