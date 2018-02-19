# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.decomposition import PCA
def load_x():
    '''
    return two:
    newDate: the data after PCA
    pca:can inverse_transform to org data
    :return:
    '''
    data=np.load('/home/cooper/Downloads/MCM/能源/num.npy')
    pca = PCA(n_components=49, whiten=True)
    newData = pca.fit_transform(data)
    print newData.shape
    print np.sum(pca.explained_variance_ratio_)
    # print pca.explained_variance_ratio_
    M = data
    print "Mshape", M.shape
    inverse = pca.inverse_transform(newData)
    emmm = np.sum(abs(M - inverse))
    return newData,pca
def load_y():
    data = np.loadtxt('/home/cooper/Downloads/MCM/buquan_exp_CSV.csv', delimiter=',', skiprows=2)
    pca = PCA(n_components=49, whiten=True)
    newData = pca.fit_transform(data)
    M = data
    print "Mshape", M.shape
    inverse = pca.inverse_transform(newData)
    emmm = np.sum(abs(M - inverse))
    return newData, pca





