# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:54:22 2018

@author: Brian
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = np.asmatrix(pd.read_excel('ozone2017.xlsx'))

l_end = len(df)

### make a tensor
a=[]
a_t=np.zeros(((12,6,4)))
a_sum = np.zeros((6,4))
a_sum_all = np.zeros(12)

count = 0

for i in range(0,l_end,6):
    
    a_t[count,:,:] = df[i:i+6,:]
    count += 1

for i in range(len(a)):
    a_sum += a_t[i,:,:]

a_ave = a_t.mean(axis=0)
a_std = a_t.std(axis=0)

#### set grid location for time series
x_long = 2
y_lat = 3

t = np.arange(0,12)
d = a_t[:,y_lat,x_long]

fig, ax = plt.subplots()
ax.plot(t, d)

plot_title = 'Ozone 2017  ' + str(x_long)+','+str(y_lat)

ax.set(xlabel='Month', ylabel='Dobson Units',
       title= plot_title)

ax.grid()

plt.show()

ax = plt.axes()
sns.heatmap(a_ave, annot=False,  linewidths=.1, 
            cbar_kws={'label': 'Dobson Units'},
            ax = ax)
ax.set_title('Average Ozone')
plt.show()

ax = plt.axes()
sns.heatmap(a_std, annot= False,  linewidths=.1, 
            cbar_kws={'label': 'Dobson Units'},
            ax = ax)
ax.set_title('Std Dev Ozone')
plt.show()

for i in range(12):
    a_sum_all[i] += a_t[i,:,:].mean()
    
fig, ax = plt.subplots()
ax.plot(t, a_sum_all)

plot_title = 'Ozone 2017 national average'

ax.set(xlabel='Month', ylabel='Dobson Units',
       title= plot_title)

ax.grid()

plt.show()
    



    



        
    