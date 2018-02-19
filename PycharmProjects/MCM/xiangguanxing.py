# -*- coding: UTF-8 -*-
import numpy as np
import pandas
import math
MN_info = pandas.read_csv('/home/cooper/Downloads/MCM/emer/MN.csv')
MN=MN_info.head(51)
mnper=MN.corr()
print mnper
mnper.to_csv('/home/cooper/Downloads/MCM/emer/mn.csv')

for index, row in mnper.iterrows():  # 获取每行的index、row
    t = 0
    for col_name in mnper.columns:
        t=t+abs(mnper[index][col_name])
    if t/28>0.651:
        print index,t/28