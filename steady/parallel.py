#!/usr/bin/python3
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy.stats import f
clf = linear_model.LinearRegression()
 
sns.set()
 
#freq = 10.0 # 年間観測回数
#width = 1.0 # 関数のSSE期間の幅
# ステップ量の検定実行
N = 1000 # 試行回数
std = 2.0 # 観測値のばらつき（標準偏差）
#irange=[1,1/2,1/4,1/6,1/8,1/10,1/12,1/25,1/50,1/100,1/365]
irange=[1/6,1/8,1/10,1/12,1/25,1/50,1/100,1/365]
diff=[2,1.5,1.25,1.0,0.75,0.5,0.4,0.3,0.2,0.1]
year = 12 #観測期間
rans = 0 # 乱数スイッチ

for dd in range(10):
 for istd in range(2,3):
  std=float(istd)
  for rans in range(1):
   for inte in irange:
    for yloopb in reversed(range(2,year,2)):
     yloop=yloopb*0.5 
     Af=[]
     for loop in range(N):
      ran=[]
      for i in range(len(np.arange(0,yloop,inte))):
       ran.append(rand()*yloop)
      tim =np.arange(0,yloop,inte)*(1-rans)#+np.sort(ran)*rans
      randi=randn(len(np.arange(0,yloop,inte)))*std
      randi2=randn(len(np.arange(0,yloop,inte)))*std +tim*std*diff[dd]
      atim=np.average(tim)
      aran=np.average(randi)
      aran2=np.average(randi2)
      b, a = np.polyfit(tim,randi,1)
      b2, a2 = np.polyfit(tim,randi2,1)
      sxy1=0
      sxx1=0
      sxy2=0
      sxx2=0
      se11=0
      se12=0
      se21=0
      se22=0
      for i in range(len(tim)):
        sxy1+=(tim[i]-atim)*(randi[i]-aran)
        sxx1+=(tim[i]-atim)*(tim[i]-atim)
        sxy2+=(tim[i]-atim)*(randi2[i]-aran2)
        sxx2+=(tim[i]-atim)*(tim[i]-atim)
      sb=(sxy1+sxy2)/(sxx1+sxx2)
      sa=aran-sb*atim
      sa2=aran2-sb*atim
      for i in range(len(tim)):
        se11+=(b*tim[i]+a-randi[i])**2
        se12+=(b2*tim[i]+a2-randi2[i])**2
        se21+=(sb*tim[i]+sa-randi[i])**2
        se22+=(sb*tim[i]+sa2-randi2[i])**2
#      print(a,b,a2,b2,sa,sb,sa2,sb,se11,se12,se21,se22)
      msa=(se21+se22-se11-se12)/(1)#自由度差１
      msb=(se11+se12)/(len(tim)*2-4)#自由度2+2
      F=msa/msb
#      print(a,b,a2,b2,msa,msb,F,f.ppf(0.05, len(tim)*2-3, len(tim)*2-4))
      if(F<f.ppf(0.05, len(tim)*2-3, len(tim)*2-4)):
        Af.append(F)
     print(rans,1/inte,yloop,diff[dd],len(Af))
#     print(Af)

