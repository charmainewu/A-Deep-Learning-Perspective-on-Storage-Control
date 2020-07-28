# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:48:54 2020

@author: jmwu
"""

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

df = pd.read_csv('simulation.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M')
df.index = df['Date']
data = df.sort_index(ascending=True, axis=0)
data = data.fillna(0)

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

renewable_data = renewable_data["2016-01-01 00:00:00":"2018-12-31 23:00:00"]
price_data = price_data["2016-01-01 00:00:00":"2018-12-31 23:00:00"]
load_data = load_data["2016-01-01 00:00:00":"2018-12-31 23:00:00"]
hour_data = hour_data["2016-01-01 00:00:00":"2018-12-31 23:00:00"]
month_data = month_data["2016-01-01 00:00:00":"2018-12-31 23:00:00"]
week_data = week_data["2016-01-01 00:00:00":"2018-12-31 23:00:00"]

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

###############################################################################
dataset_p  = np.hstack((p,w,h))
dataset_r  = np.hstack((r,w,h))
for n_month in [1,3,6,12]:
    for order in range(1,36):
        try:
            start_point = order*n_month*int(len(dataset_p)*0.027)
            train_size = int(len(dataset_p)*n_month*0.027)
            test_size = int(len(dataset_p)*n_month*0.027)
            train, test = dataset_p[start_point:start_point+train_size,:], dataset_p[start_point+train_size:start_point+train_size+test_size,:]
            
            BACK_DAY = 24
            x_train, y_train = [], []
            x_test, y_test = [], []
            
            for i in range(BACK_DAY,train_size):
                x_train.append(train[i-BACK_DAY:i,:])
                y_train.append(train[i,0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            
            for i in range(BACK_DAY,test_size):
                x_test.append(test[i-BACK_DAY:i,:])
                y_test.append(test[i,0])
            x_test, y_test = np.array(x_test), np.array(y_test)
            
            model = Sequential()
            model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(units=128))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(x_train, y_train, epochs=200, batch_size= 100, validation_data=(x_test, y_test), verbose=2)
            
            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)
            train_predict = scaler_p.inverse_transform(train_predict)
            y_train = scaler_p.inverse_transform([y_train])
            test_predict = scaler_p.inverse_transform(test_predict)
            y_test = scaler_p.inverse_transform([y_test])
            
            model.save("./pricedata/pointp"+str(order)+str(n_month)+".h5")
            del model
            print('Train Mean Absolute Error:', mean_absolute_error(y_train[0], train_predict[:,0]))
            print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train[0], train_predict[:,0])))
            print('Test Mean Absolute Error:', mean_absolute_error(y_test[0], test_predict[:,0]))
            print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[0], test_predict[:,0])))
            
            aa=[x for x in range(200)]
            plt.figure(figsize=(8,4))
            plt.plot(aa, y_test[0][:200], 'tab:green', marker='.', label="actual")
            plt.plot(aa, test_predict[:,0][:200], 'tab:orange', label="prediction")
            plt.tight_layout()
            sns.despine(top=True)
            plt.subplots_adjust(left=0.07)
            plt.ylabel('Price', size=15)
            plt.xlabel('Time step', size=15)
            plt.legend(fontsize=15)
            plt.show()
            
            start_point = order*n_month*int(len(dataset_p)*0.027)
            train_size = int(len(dataset_p)*n_month*0.027)
            test_size = int(len(dataset_p)*n_month*0.027)
            train, test = dataset_p[start_point:start_point+train_size,:], dataset_p[start_point+train_size:start_point+train_size+test_size,:]
            
            BACK_DAY = 24
            x_train, y_train = [], []
            x_test, y_test = [], []
            
            for i in range(BACK_DAY,train_size):
                x_train.append(train[i-BACK_DAY:i,:])
                y_train.append(train[i,0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            
            for i in range(BACK_DAY,test_size):
                x_test.append(test[i-BACK_DAY:i,:])
                y_test.append(test[i,0])
            x_test, y_test = np.array(x_test), np.array(y_test)
            
            model = Sequential()
            model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(units=128))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(x_train, y_train, epochs=200, batch_size=100, validation_data=(x_test, y_test), verbose=2)
            
            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)
            train_predict = scaler_r.inverse_transform(train_predict)
            y_train = scaler_r.inverse_transform([y_train])
            test_predict = scaler_r.inverse_transform(test_predict)
            y_test = scaler_r.inverse_transform([y_test])
            
            model.save("./winddata/pointr"+str(order)+str(n_month)+".h5")
            del model
            
            print('Train Mean Absolute Error:', mean_absolute_error(y_train[0], train_predict[:,0]))
            print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train[0], train_predict[:,0])))
            print('Test Mean Absolute Error:', mean_absolute_error(y_test[0], test_predict[:,0]))
            print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[0], test_predict[:,0])))
            
            aa=[x for x in range(200)]
            plt.figure(figsize=(8,4))
            plt.plot(aa, y_test[0][:200], 'tab:green', marker='.', label="actual")
            plt.plot(aa, test_predict[:,0][:200], 'tab:orange', label="prediction")
            #plt.tight_layout()
            sns.despine(top=True)
            plt.subplots_adjust(left=0.07)
            plt.ylabel('Renewables', size=15)
            plt.xlabel('Time step', size=15)
            plt.legend(fontsize=15)
            plt.show()
        except:
            continue