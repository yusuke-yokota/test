#!/usr/bin/python3
#%matplotlib inline
import numpy as np
from scipy import stats
from numpy.random import *

sigma = 1.5
#step  = 3.0
#dat   = 5

path_w = 'test'
#for dat in range(2,10):
for dat in range(365,366):
  for step_f in range(41):
   p_c=[0,0,0,0]
   p_t=[0,0,0,0]
#   step = 0.0 + step_f * 0.2
   step = 0.0 + step_f * 0.1
   for loop in range(10000):
#    file = open(path_w+'_'+str(dat)+'_'+str(step_f)+'.dat', mode='w')
    A = randn(dat)
    B = randn(dat) + (step/sigma)
    res = stats.ttest_ind(A, B).pvalue
#res = stats.ttest_ind(A, B).statistic
#print(res)
    if(res<0.01):
      p_c[0] = p_c[0]+1
    elif(res<0.05):
      p_c[1] = p_c[1]+1
    elif(res<0.1):
      p_c[2] = p_c[2]+1
    else:
      p_c[3] = p_c[3]+1
    ret = stats.ttest_rel(A, B).pvalue
#res = stats.ttest_rel(A, B).statistic
#print(res)
    if(ret<0.01):
      p_t[0] = p_t[0]+1
    elif(ret<0.05):
      p_t[1] = p_t[1]+1
    elif(ret<0.1):
      p_t[2] = p_t[2]+1
    else:
      p_t[3] = p_t[3]+1

#    file.write(str(res)+", "+str(p_c)+", "+str(ret)+", "+str(p_t))
#    file.close()
   print(dat,step,p_c[0],p_c[0]+p_c[1],p_c[0]+p_c[1]+p_c[2],p_c[3],p_t[0],p_t[0]+p_t[1],p_t[0]+p_t[1]+p_t[2],p_t[3])
