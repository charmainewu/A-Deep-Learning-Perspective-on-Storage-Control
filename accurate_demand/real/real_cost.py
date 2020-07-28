# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:45:11 2020

@author: jmwu
"""

from benchmark import BenchmarkAccurateDemand
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import time
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import seaborn as sns

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
for i in range(0,len(data)):
    hour_data['Date'][i] = data['Date'][i]
    if data['Date'][i].hour<6:
        hour_data['Hour'][i] = 1
    elif data['Date'][i].hour>=6 and data['Date'][i].hour<12:
        hour_data['Hour'][i] = 2
    elif data['Date'][i].hour>=12 and data['Date'][i].hour<18:
        hour_data['Hour'][i] = 3
    else:
        hour_data['Hour'][i] = 4
hour_data.index = hour_data.Date
hour_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    month_data['Date'][i] = data['Date'][i]
    month_data['Month'][i] = data['Date'][i].month
month_data.index = month_data.Date
month_data.drop('Date', axis=1, inplace=True)

for i in range(0,len(data)):
    week_data['Date'][i] = data['Date'][i]
    week_data['Week'][i] = data['Date'][i].weekday()
week_data.index = week_data.Date
week_data.drop('Date', axis=1, inplace=True)

renewable_data = renewable_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
price_data = price_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
load_data = load_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
hour_data = hour_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
month_data = month_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
week_data = week_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]

scaler_r = MinMaxScaler(feature_range=(0, 1))
r = scaler_r.fit_transform(renewable_data)

scaler_p = MinMaxScaler(feature_range=(0, 1))
p = scaler_p.fit_transform(price_data)

enc_h = OneHotEncoder(handle_unknown='ignore')
h = enc_h.fit_transform(hour_data).toarray()

enc_m = OneHotEncoder(handle_unknown='ignore')
m = enc_m.fit_transform(month_data).toarray()

enc_w = OneHotEncoder(handle_unknown='ignore')
w = enc_w.fit_transform(week_data).toarray()

dataset_r  = np.hstack((r,w,h))
dataset_p  = np.hstack((p,w,h))

###########################################################################

Demand = load_data.values.reshape(len(load_data))
Price = price_data.values.reshape(len(price_data))
Renewables = renewable_data.values.reshape(len(renewable_data))
PriceScalar = scaler_p
WindScalar = scaler_r
WindStackData = dataset_r
PriceStackData = dataset_p

Interval = 100; 
Window= 50; 
Capacity = 100000; 
Ntrain = int(len(dataset_r) * 0.50)
Nvalid = int(len(dataset_r) * 0.30)
Ntest = int(len(dataset_r) * 0.20)
Backday = 12;
##########################################################
LSTM = "Point"
WindModel = load_model("./winddata/point.h5")
PriceModel = load_model("./pricedata/point.h5")

BAD = BenchmarkAccurateDemand(Capacity, Window, Ntrain, Ntest, Nvalid, Backday, Interval, Price, Demand, Renewables, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM)
cost_deta= BAD.deta()
cost_mpc= BAD.mpc()
cost_rl= BAD.rl()
cost_thb= BAD.thb()
cost_peta = BAD.leta()
cost_ofl = BAD.ofl()
cost_nos = BAD.nos()
print("end")
##########################################################

LSTM = "S2S"
WindModel = Seq2Seq(output_dim=1, hidden_dim=64, output_length=10, input_shape=(Backday, dataset_r.shape[1]), peek=False, depth=2, dropout = 0.2)

PriceModel = Seq2Seq(output_dim=1, hidden_dim=64, output_length=10, input_shape=(Backday, dataset_r.shape[1]), peek=False, depth=2, dropout = 0.2)

WindModel.load_weights("./winddata/s2s10.h5")
PriceModel.load_weights("./pricedata/s2s10.h5")


BAD = BenchmarkAccurateDemand(Capacity, Window, Ntrain, Ntest, Nvalid, Backday, Interval, Price, Demand, Renewables, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM)
cost_seta = BAD.leta()
print("end")
##########################################################
LSTM = "Vector"
WindModel = load_model("./winddata/vector10.h5")
PriceModel = load_model("./pricedata/vector10.h5")

BAD = BenchmarkAccurateDemand(Capacity, Window, Ntrain, Ntest, Nvalid, Backday, Interval, Price, Demand, Renewables, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM)
cost_veta = BAD.leta()
print("end")

np.save("./data/cost_deta.npy",cost_deta)
np.save("./data/cost_mpc.npy",cost_mpc)
np.save("./data/cost_rl.npy",cost_rl)
np.save("./data/cost_thb.npy",cost_thb)
np.save("./data/cost_ofl.npy",cost_ofl)
np.save("./data/cost_nos.npy",cost_nos)
np.save("./data/cost_peta.npy",cost_peta)
np.save("./data/cost_veta.npy",cost_veta)
np.save("./data/cost_seta.npy",cost_seta)

