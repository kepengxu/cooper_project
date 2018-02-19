# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
#filename  = '/home/cooper/Downloads/MCM/能源/亚里桑那'
#fo = open('/home/cooper/Downloads/MCM/能源/亚里桑那.txt', 'w')
with open('/home/cooper/Downloads/MCM/能源/加州.txt', 'w') as fp:
    for line in open('/home/cooper/Downloads/MCM/能源/加州'):
        l=str(line)
        l = l.replace('年', '')
        l = l.replace(',', '')
        l = l.replace('N', '')
        l = l.replace('A', '0')
        print l
        fp.write(l+ '\n')
#        print line
#        print line[16]
#        j=-1
#        for i in line:
#            j = j + 1
#            print j,line[j]
#
#
#        line.lstrip(',')
#        print line
#        break
#        fp.write(line+'\n')
#        fp.write(line.lstrip(',')+"\n")
print 'ok'
data2=np.loadtxt('/home/cooper/Downloads/MCM/能源/加州.txt')
print data2.shape