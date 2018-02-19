# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
from sklearn.decomposition import PCA

def load_x():
    data=np.loadtxt('/home/cooper/Downloads/MCM/emer/MN.csv',skiprows=1,delimiter=',')
    print "data shape",data.shape
    pca = PCA(n_components=5)
    newData = pca.fit_transform(data)
    print newData.shape
    print np.sum(pca.explained_variance_ratio_)
    # print pca.explained_variance_ratio_
    M = data
    print "Mshape", M.shape
    inverse = pca.inverse_transform(newData)
    emmm = np.sum(abs(M - inverse))
    return newData,data,pca
newdata,data,pca=load_x()
newdata=newdata.reshape((50,5))
m = 1998 - 1960
f = np.zeros((m, 1))
total = f
for i in range(5):

    history_data = newdata[:,i:i+1]
    n = len(history_data)
    X0 = np.array(history_data)

    # 累加生成
    history_data_agg = [sum(history_data[0:i + 1]) for i in range(n)]
    X1 = np.array(history_data_agg)

    # 计算数据矩阵B和数据向量Y
    B = np.zeros([n - 1, 2])
    Y = np.zeros([n - 1, 1])
    for i in range(0, n - 1):
        B[i][0] = -0.5 * (X1[i] + X1[i + 1])
        B[i][1] = 1
        Y[i][0] = X0[i + 1]

    # 计算GM(1,1)微分方程的参数a和u
    # A = np.zeros([2,1])
    A = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    a = A[0][0]
    u = A[1][0]

    # 建立灰色预测模型
    XX0 = np.zeros(n)
    XX0[0] = X0[0]
    for i in range(1, n):
        XX0[i] = (X0[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * (i));

    # 模型精度的后验差检验
    e = 0  # 求残差平均值
    for i in range(0, n):
        e += (X0[i] - XX0[i])
    e /= n

    # 求历史数据平均值
    aver = 0;
    for i in range(0, n):
        aver += X0[i]
    aver /= n

    # 求历史数据方差
    s12 = 0;
    for i in range(0, n):
        s12 += (X0[i] - aver) ** 2;
    s12 /= n

    # 求残差方差
    s22 = 0;
    for i in range(0, n):
        s22 += ((X0[i] - XX0[i]) - e) ** 2;
    s22 /= n

    # 求后验差比值
    C = s22 / s12

    # 求小误差概率
    cout = 0
    for i in range(0, n):
        if abs((X0[i] - XX0[i]) - e) < 0.6754 * math.sqrt(s12):
            cout = cout + 1
        else:
            cout = cout
    P = cout / n

    if True:#(C < 0.35 and P > 0.95):
        # 预测精度为一级
           # 请输入需要预测的年数
        # print('往后m各年负荷为：')

        for i in range(0, m):
            f[i] = (X0[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * (i + n))
        total=np.column_stack((total,f))
    else:
        print('灰色预测法不适用')
print total[:,0]
total=total[:,1:6]
print 'total shape',total.shape
after=pca.inverse_transform(total)