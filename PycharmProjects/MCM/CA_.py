# -*- coding: UTF-8 -*-
import numpy as np
import pandas
import math
dataMN=np.loadtxt('/home/cooper/Downloads/MCM/emer/MN.csv',skiprows=1,delimiter=',')
dataCA=np.loadtxt('/home/cooper/Downloads/MCM/emer/CA.csv',skiprows=1,delimiter=',')
dataTX=np.loadtxt('/home/cooper/Downloads/MCM/emer/TX.csv',skiprows=1,delimiter=',')
dataAZ=np.loadtxt('/home/cooper/Downloads/MCM/emer/AZ.csv',skiprows=1,delimiter=',')
MN_info = pandas.read_csv('/home/cooper/Downloads/MCM/emer/MN.csv')
MN=MN_info.head(0).columns.tolist()
CA_info = pandas.read_csv('/home/cooper/Downloads/MCM/emer/CA.csv')
CA=CA_info.head(0).columns.tolist()
TX_info = pandas.read_csv('/home/cooper/Downloads/MCM/emer/TX.csv')
TX=TX_info.head(0).columns.tolist()
AZ_info = pandas.read_csv('/home/cooper/Downloads/MCM/emer/AZ.csv')
AZ=AZ_info.head(0).columns.tolist()
#with open('/home/cooper/Downloads/MCM/emer/MN.csv') as f:
#    MN_csv = csv.reader(f)
#    headers = next(MN_csv)
print dataMN.shape,'MN'
print dataCA.shape
print dataAZ.shape
print dataTX.shape

	#计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean
#计算Pearson系数
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p
#计算每个特征的spearman系数，返回数组
def calcAttribute(dataSet):
    prr = []
    n,m = np.shape(dataSet)    #获取数据集行数和列数
    x = [0] * n             #初始化特征x和类别y向量
    y = [0] * n
    for i in range(n):      #得到类向量
        y[i] = dataSet[i][m-1]
    for j in range(m-1):    #获取每个特征的向量，并计算Pearson系数，存入到列表中
        for k in range(n):
            x[k] = dataSet[k][j]
        prr.append(calcPearson(x,y))
    return prr
mn= calcAttribute(dataMN)
paixu=mn
lmn=list()
for i in range(27):
    lmn.append({'name':MN[i],'data':mn[i]})
lmn.sort(key=lambda x:x['data'],reverse=True)
print 'MN',lmn
#MN [{'data': 0.91200357423510003, 'name': 'BMTCB'}, {'data': 0.75725797818847063, 'name': 'RETCB'}, {'data': 0.29131230012150744, 'name': 'DFTCB'}, {'data': 0.14821916033315563, 'name': 'PATCB'}
ca=calcAttribute(dataCA)
az=calcAttribute(dataAZ)
tx=calcAttribute(dataTX)


