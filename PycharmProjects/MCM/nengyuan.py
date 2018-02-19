# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
data=np.loadtxt('/home/cooper/Downloads/MCM/emer/AZ.csv',skiprows=1,delimiter=',')
total=data[:,-2]
rec=data[:,-3]
y=rec/total
year=np.arange(50)+1960
#pre=p1*year*year**2+year**2+year


