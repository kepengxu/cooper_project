# -*- coding: UTF-8 -*-
#Arizona     California   New Mexico   Texas
#shape[50,four tem + four pre +four population + four total_GDP + four manufact_GDP]
import numpy as np
data1=np.loadtxt('/home/cooper/Downloads/MCM/能源/亚里桑那.txt')

total_data=data1[:,1:13]
print data1.shape
data2=np.loadtxt('/home/cooper/Downloads/MCM/能源/加州.txt')
total_data=np.column_stack((total_data,data2[:,1:13]))

print data2.shape
data4=np.loadtxt('/home/cooper/Downloads/MCM/能源/新墨西哥.txt')
total_data=np.column_stack((total_data,data4[:,1:13]))
print data4.shape
data3=np.loadtxt('/home/cooper/Downloads/MCM/能源/德州.txt')
total_data=np.column_stack((total_data,data3[:,1:13]))
print data3.shape
print total_data.shape

def load1():
    '''
    Arizona     California   New Mexico   Texas
    :return:
    '''
    data = np.loadtxt('/home/cooper/Downloads/MCM/   California 气温.txt')
    data1 = data[0:50, 1:13]
    California_Mean_tem = np.mean(data1, axis=1).reshape((50, 1))
    total_data = California_Mean_tem
    print '加州平均气温'
    print California_Mean_tem.shape

    # 处理亚里桑那平均气温
    data = np.loadtxt('/home/cooper/Downloads/MCM/Arizona气温.txt')
    data2 = data[0:50, 1:13]
    Arizona_Mean_tem = np.mean(data2, axis=1).reshape((50, 1))
    total_data = np.column_stack((total_data, Arizona_Mean_tem))
    print '亚里桑那平均气温'
    print Arizona_Mean_tem.shape

    # 处理新墨西哥平均气温
    data = np.loadtxt('/home/cooper/Downloads/MCM/new_moxige气温.txt')
    data3 = data[0:50, 1:13]
    new_Mexico_Mean_tem = np.mean(data3, axis=1).reshape((50, 1))
    total_data = np.column_stack((total_data, new_Mexico_Mean_tem))
    print '新墨西哥平均气温'
    print new_Mexico_Mean_tem.shape

    # 处理德州平均气温
    data = np.loadtxt('/home/cooper/Downloads/MCM/德州气温.txt')
    data4 = data[0:50, 1:13]
    Texas_Mean_tem = np.mean(data4, axis=1).reshape((50, 1))
    total_data = np.column_stack((total_data, Texas_Mean_tem))
    print '德州平均气温'
    print Texas_Mean_tem.shape

    print 'temperature is ok'

    # 处理平均降水量

    # 处理加州月平均降水量
    data = np.loadtxt('/home/cooper/Downloads/MCM/precipitation_data/  California.txt')
    data11 = data[0:50, 1:13]
    California_Mean_pre = np.mean(data11, axis=1).reshape((50, 1))
    total_data = np.column_stack((total_data, California_Mean_pre))
    print '加州月平均降水量'
    print California_Mean_pre.shape

    # 处理亚利桑那月平均降水量
    data = np.loadtxt('/home/cooper/Downloads/MCM/precipitation_data/Arizona.txt')
    data12 = data[0:50, 1:13]
    Arizona_Mean_pre = np.mean(data12, axis=1).reshape((50, 1))
    total_data = np.column_stack((total_data, Arizona_Mean_pre))
    print '亚利桑那月平均降水量'
    print Arizona_Mean_pre.shape

    # 处理新墨西哥月平均降水量
    data = np.loadtxt('/home/cooper/Downloads/MCM/precipitation_data/New_Mexico.txt')
    data13 = data[0:50, 1:13]
    New_Mexico_Mean_pre = np.mean(data13, axis=1).reshape((50, 1))
    total_data = np.column_stack((total_data, New_Mexico_Mean_pre))
    print '新墨西哥月平均降水量'
    print New_Mexico_Mean_pre.shape

    # 处理德州月平均降水量
    data = np.loadtxt('/home/cooper/Downloads/MCM/precipitation_data/Texas.txt')
    data14 = data[0:50, 1:13]
    Texas_Mean_pre = np.mean(data14, axis=1).reshape((50, 1))
    total_data = np.column_stack((total_data, Texas_Mean_pre))
    print '德州月平均降水量'
    print Texas_Mean_pre.shape

    print 'precipitation_data is ok'

    # 人口
    data = np.loadtxt('/home/cooper/Downloads/MCM/population.csv', delimiter=',')
    population_data = data.transpose()
    total_data = np.column_stack((total_data, population_data))
    print 'population'
    print population_data.shape

    # total GDP
    data = np.loadtxt('/home/cooper/Downloads/MCM/data_GDP.csv', delimiter=',')
    GDP_data = data[1:5, ].transpose()
    total_data = np.column_stack((total_data, GDP_data))
    print GDP_data.shape

    # Manufacturing data
    data = np.loadtxt('/home/cooper/Downloads/MCM/data_GDP.csv', delimiter=',')
    manu_GDP_data = data[5:9, ].transpose()
    total_data = np.column_stack((total_data, manu_GDP_data))
    print manu_GDP_data.shape
    print total_data.shape
    return total_data
total_data=np.column_stack((total_data,load1()))
print total_data.shape
np.save('/home/cooper/Downloads/MCM/能源/num.npy',total_data)