from sklearn.decomposition import PCA
import numpy as np
def load_x(file_path):
    data=np.loadtxt(file_path,skiprows=1,delimiter=',')
   # print "data shape",data.shape
    pca = PCA(n_components=4)
    newData = pca.fit_transform(data)
    #print newData.shape
    #print np.sum(pca.explained_variance_ratio_)
    # print pca.explained_variance_ratio_
    M = data
    #print "Mshape", M.shape
    inverse = pca.inverse_transform(newData)
    emmm = np.sum(abs(M - inverse))
    return newData,data,pca

a,data,b=load_x('/home/cooper/Downloads/MCM/emer/MN.csv')
print 'MN',b.explained_variance_ratio_
a,data,b=load_x('/home/cooper/Downloads/MCM/emer/CA.csv')
print 'CA',b.explained_variance_ratio_
a,data,b=load_x('/home/cooper/Downloads/MCM/emer/TX.csv')
print 'TX',b.explained_variance_ratio_
a,data,b=load_x('/home/cooper/Downloads/MCM/emer/AZ.csv')
print 'AZ',b.explained_variance_ratio_
np.savetxt('/home/cooper/Downloads/MCM/emer/MN_PCA.csv', a, delimiter=',')
#datamat=np.mat(data)
#print datamat.dtype
#newmat=np.mat(a)

#ni=datamat.I
#transf=ni*newmat
#print transf
#np.savetxt('/home/cooper/Downloads/MCM/emer/TX_PCA_transform_mat.csv',transf,delimiter=',')
#print tran.shape