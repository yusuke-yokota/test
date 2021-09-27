#!/usr/bin/python3
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import *
from scipy.interpolate import interp1d
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
clf = linear_model.LinearRegression()
 
sns.set()
 
#freq = 10.0 # 年間観測回数
#width = 1.0 # 関数のSSE期間の幅
# ステップ量の検定実行
df = pd.read_csv('test.dat', delim_whitespace=True)
#print(df.x1)
fig=plt.figure(figsize=(25,5))
ax1=fig.add_subplot(1,5,1)
df_0=df[(df.r == 0) & (df.freq == 50.)]
df_1=df[(df.r == 1) & (df.freq == 50.)]
ax1.set_ylim(0,2)
#ax1.plot(df_0.year,df_0.x2,marker="o")
#ax1.plot(df_1.year,df_1.x2,marker="o")
ax1.scatter(df_0.year,df_0.x2)

ax2=fig.add_subplot(1,5,2)
df_0=df[(df.r == 0) & (df.freq == 25.)]
df_1=df[(df.r == 1) & (df.freq == 25.)]
ax2.set_ylim(0,2)
#ax2.plot(df_0.year,df_0.x2,marker="o")
#ax2.plot(df_1.year,df_1.x2,marker="o")
ax2.scatter(df_0.year,df_0.x2)

ax3=fig.add_subplot(1,5,3)
df_0=df[(df.r == 0) & (df.freq == 12.)]
df_1=df[(df.r == 1) & (df.freq == 12.)]
ax3.set_ylim(0,2)
#ax3.plot(df_0.year,df_0.x2,marker="o")
#ax3.plot(df_1.year,df_1.x2,marker="o")
ax3.scatter(df_0.year,df_0.x2)

ax4=fig.add_subplot(1,5,4)
df_0=df[(df.r == 0) & (df.freq == 6.)]
df_1=df[(df.r == 1) & (df.freq == 6.)]
ax4.set_ylim(0,2)
ax4.scatter(df_0.year,df_0.x2)

ax5=fig.add_subplot(1,5,5)
df_0=df[(df.r == 0) & (df.freq == 4.)]
df_1=df[(df.r == 1) & (df.freq == 4.)]
ax5.set_ylim(0,2)
ax5.scatter(df_0.year,df_0.x2)
fig.savefig('test.pdf')

fig=plt.figure(figsize=(10,5))
ax1=fig.add_subplot(1,2,1)
df_0=df[(df.r == 0) & (df.freq == 4.)]
df_1=df[(df.r == 1) & (df.freq == 4.)]
ax1.set_ylim(0,2)
ax1.scatter(df_0.year,df_0.x2)
ax1.scatter(df_1.year,df_1.x2)
fig.savefig('test0.pdf')

fig=plt.figure(figsize=(25,5))
ax1=fig.add_subplot(1,5,1)
df_0=df[(df.r == 0) & (df.year == 5.)]
df_1=df[(df.r == 1) & (df.year == 5.)]
ax1.set_ylim(0,2)
ax1.scatter(df_0.freq,df_0.x2)

ax2=fig.add_subplot(1,5,2)
df_0=df[(df.r == 0) & (df.year == 4.)]
df_1=df[(df.r == 1) & (df.year == 4.)]
ax2.set_ylim(0,2)
ax2.scatter(df_0.freq,df_0.x2)

ax3=fig.add_subplot(1,5,3)
df_0=df[(df.r == 0) & (df.year == 3.)]
df_1=df[(df.r == 1) & (df.year == 3.)]
ax3.set_ylim(0,2)
ax3.scatter(df_0.freq,df_0.x2)

ax4=fig.add_subplot(1,5,4)
df_0=df[(df.r == 0) & (df.year == 2.)]
df_1=df[(df.r == 1) & (df.year == 2.)]
ax4.set_ylim(0,2)
ax4.scatter(df_0.freq,df_0.x2)

ax5=fig.add_subplot(1,5,5)
df_0=df[(df.r == 0) & (df.year == 1.)]
df_1=df[(df.r == 1) & (df.year == 1.)]
ax5.set_ylim(0,2)
ax5.scatter(df_0.freq,df_0.x2)

fig.savefig('test1.pdf')

fig=plt.figure(figsize=(15,5))
ax1=fig.add_subplot(1,3,1)
ax1.set_ylim(0,2)
df_0=df[(df.r == 0) & (df.freq == 50.)]
df_1=df[(df.r == 0) & (df.freq == 25.)]
df_2=df[(df.r == 0) & (df.freq == 12.)]
df_3=df[(df.r == 0) & (df.freq == 6.)]
df_4=df[(df.r == 0) & (df.freq == 4.)]
df_5=df[(df.r == 0) & (df.freq == 2.)]
df_6=df[(df.r == 0) & (df.freq == 1.)]
df_X=df[(df.r == 0) & (df.freq == 365.)]
out_x = np.linspace(np.min(df_0.year), np.max(df_0.year), np.size(df_0.year)*100)
func_spline = interp1d(df_0.year, df_0.x2, kind='cubic')
out_y = func_spline(out_x)
ax1.plot(out_x,out_y)
ax1.scatter(df_0.year,df_0.x2)
out_x = np.linspace(np.min(df_1.year), np.max(df_1.year), np.size(df_1.year)*100)
func_spline = interp1d(df_1.year, df_1.x2, kind='cubic')
out_y = func_spline(out_x)
ax1.plot(out_x,out_y)
ax1.scatter(df_1.year,df_1.x2)
out_x = np.linspace(np.min(df_2.year), np.max(df_2.year), np.size(df_2.year)*100)
func_spline = interp1d(df_2.year, df_2.x2, kind='cubic')
out_y = func_spline(out_x)
ax1.plot(out_x,out_y)
ax1.scatter(df_2.year,df_2.x2)
out_x = np.linspace(np.min(df_3.year), np.max(df_3.year), np.size(df_3.year)*100)
func_spline = interp1d(df_3.year, df_3.x2, kind='cubic')
out_y = func_spline(out_x)
ax1.plot(out_x,out_y)
ax1.scatter(df_3.year,df_3.x2)
out_x = np.linspace(np.min(df_4.year), np.max(df_4.year), np.size(df_4.year)*100)
func_spline = interp1d(df_4.year, df_4.x2, kind='cubic')
out_y = func_spline(out_x)
ax1.plot(out_x,out_y)
ax1.scatter(df_4.year,df_4.x2)
out_x = np.linspace(np.min(df_5.year), np.max(df_5.year), np.size(df_5.year)*100)
func_spline = interp1d(df_5.year, df_5.x2, kind='cubic')
out_y = func_spline(out_x)
ax1.plot(out_x,out_y)
ax1.scatter(df_5.year,df_5.x2)
out_x = np.linspace(np.min(df_6.year), np.max(df_6.year), np.size(df_6.year)*100)
func_spline = interp1d(df_6.year, df_6.x2, kind='cubic')
out_y = func_spline(out_x)
ax1.plot(out_x,out_y)
ax1.scatter(df_6.year,df_6.x2)
out_x = np.linspace(np.min(df_X.year), np.max(df_X.year), np.size(df_X.year)*100)
func_spline = interp1d(df_X.year, df_X.x2, kind='cubic')
out_y = func_spline(out_x)
ax1.plot(out_x,out_y)
ax1.scatter(df_X.year,df_X.x2)
ax1.set_xscale('log')
fig.savefig('test2.pdf')
##########################
ax2=fig.add_subplot(1,3,2)
ax2.set_ylim(0,2)
ax2.set_xlim(0.5,400)
ax2.set_xticks([1,2,4,6,8,10,12,25,50,100,365])
ax2.set_xscale('log')
df_0=df[(df.r == 0) & (df.year == 5.)]
ax2.scatter(df_0.freq,df_0.x2)
df_1=df[(df.r == 0) & (df.year == 4.)]
ax2.scatter(df_0.freq,df_1.x2)
df_2=df[(df.r == 0) & (df.year == 3.)]
ax2.scatter(df_0.freq,df_2.x2)
df_3=df[(df.r == 0) & (df.year == 2.)]
ax2.scatter(df_0.freq,df_3.x2)
df_4=df[(df.r == 0) & (df.year == 1.)]
ax2.scatter(df_4.freq,df_4.x2)
fig.savefig('test3.pdf')
##########################
t_95=[999,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131,2.120,2.101,2.093,2.086,2.]
ax3=fig.add_subplot(1,3,3)
ax3.set_ylim(0,4)
ax3.set_xlim(0.5,400)
ax3.set_xticks([1,2,4,6,8,10,12,25,50,100,365])
ax3.set_xscale('log')
df_0=df[(df.r == 0) & (df.year == 5.)]
df_1=df[(df.r == 0) & (df.year == 4.)]
df_2=df[(df.r == 0) & (df.year == 3.)]
df_3=df[(df.r == 0) & (df.year == 2.)]
df_4=df[(df.r == 0) & (df.year == 1.)]
df_0=df_0.reset_index()
df_1=df_1.reset_index()
df_2=df_2.reset_index()
df_3=df_3.reset_index()
df_4=df_4.reset_index()
for i in range(len(df_0)):
	if (df_0.freq[i]*5-2) > 20:
		df_0.x2[i]=df_0.x2[i]*2.
	else:
		df_0.x2[i]=df_0.x2[i]*t_95[int(df_0.freq[i]*5-2)]
for i in range(len(df_1)):
	if (df_1.freq[i]*4-2) > 20:
		df_1.x2[i]=df_1.x2[i]*2.
	else:
		df_1.x2[i]=df_1.x2[i]*t_95[int(df_1.freq[i]*4-2)]
for i in range(len(df_2)):
	if (df_2.freq[i]*3-2) > 20:
		df_2.x2[i]=df_2.x2[i]*2.
	else:
		df_2.x2[i]=df_2.x2[i]*t_95[int(df_2.freq[i]*3-2)]
for i in range(len(df_3)):
	if (df_3.freq[i]*2-2) > 20:
		df_3.x2[i]=df_3.x2[i]*2.
	else:
		df_3.x2[i]=df_3.x2[i]*t_95[int(df_3.freq[i]*2-2)]
for i in range(len(df_4)):
	if (df_4.freq[i]*1-2) > 20:
		df_4.x2[i]=df_4.x2[i]*2.
	elif (df_4.freq[i]*1-2) < 0:
		df_4.x2[i]=df_4.x2[i]*9999.
	else:
		df_4.x2[i]=df_4.x2[i]*t_95[int(df_4.freq[i]*1-2)]
ax3.scatter(df_0.freq,df_0.x2)
ax3.scatter(df_0.freq,df_1.x2)
ax3.scatter(df_0.freq,df_2.x2)
ax3.scatter(df_0.freq,df_3.x2)
ax3.scatter(df_4.freq,df_4.x2)
fig.savefig('test3.pdf')
