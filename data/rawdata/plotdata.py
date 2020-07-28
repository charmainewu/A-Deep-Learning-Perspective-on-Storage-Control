# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:24:50 2020

@author: jmwu
"""

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt


df = pd.read_csv('simulation.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M')
df.index = df['Date']
data = df.sort_index(ascending=True, axis=0)

price_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Price'])
renewable_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Wind'])
load_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Load'])
hour_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Hour'])
week_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Week'])
month_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Month'])

for i in range(0,len(data)):
    price_data['Date'][i] = data['Date'][i]
    price_data['Price'][i] = max(0,data['Price'][i])
price_data.index = price_data.Date
price_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    renewable_data['Date'][i] = data['Date'][i]
    renewable_data['Wind'][i] = data['Wind'][i]
renewable_data.index = renewable_data.Date
renewable_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    load_data['Date'][i] = data['Date'][i]
    load_data['Load'][i] = data['Load'][i]
load_data.index = load_data.Date
load_data.drop('Date', axis=1, inplace=True)

renewable_data = renewable_data["2018-01-01 00:00:00":"2018-01-31 23:00:00"]
price_data = price_data["2018-01-01 00:00:00":"2018-01-31 23:00:00"]
load_data = load_data["2018-01-01 00:00:00":"2018-01-31 23:00:00"]

plt.figure(figsize=(20,13))
ax1 = plt.subplot(2,1,1)
ax2 = ax1.twinx()
ax1.plot(price_data, linewidth = 4, label='Price Series',c = 'royalblue')
ax1.set_ylabel('Price \N{euro sign}/MW',fontsize=30)
ax1.tick_params(axis="x",labelsize=21)
ax1.tick_params(axis="y",labelsize=21)
ax1.set_ylim(0,100)
ax1.legend(fontsize=30)
ax2.plot(load_data/10000, linewidth = 4,label='Demand Series',c = 'gold')
ax2.set_ylabel('Demand/10000 MW',fontsize=30)
ax2.legend(fontsize=30,loc=2)
ax2.tick_params(axis="x",labelsize=21)
ax2.tick_params(axis="y",labelsize=21)
ax2.set_ylim(0,10)

ax1 = plt.subplot(2,1,2)
ax1.plot(renewable_data/10000,linewidth = 4, label='Renewables Series',c = 'orange')
ax1.set_ylabel('Renewables/10000 MW',fontsize=30)
ax1.set_xlabel('Time/h',fontsize=30)
ax1.legend(fontsize=30,loc=2)
ax1.tick_params(axis="x",labelsize=21)
ax1.tick_params(axis="y",labelsize=21)
plt.savefig("data.pdf")