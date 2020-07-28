# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:42:39 2020

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
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from keras.models import load_model

##################################read data####################################
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

##################################renewable pre backday = 6####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 6; n_outputs = 10; hidden = 64
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
model.add(LSTM(units=hidden, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/step/6/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/step/6/singlemae.npy",mae)
np.save("./pricedata/step/6/singlermse.npy",rmse)

##################################renewable pre backday = 12####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 12; n_outputs = 10; hidden = 64
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
model.add(LSTM(units=hidden, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/step/12/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/step/12/singlemae.npy",mae)
np.save("./pricedata/step/12/singlermse.npy",rmse)

##################################renewable pre backday = 24####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 24; n_outputs = 10; hidden = 64
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
model.add(LSTM(units=hidden, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/step/24/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/step/24/singlemae.npy",mae)
np.save("./pricedata/step/24/singlermse.npy",rmse)

##################################renewable pre state = 16####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 12; n_outputs = 10; hidden = 16
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
model.add(LSTM(units=hidden, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/state/16/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/state/16/singlemae.npy",mae)
np.save("./pricedata/state/16/singlermse.npy",rmse)

##################################renewable pre state = 64####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 12; n_outputs = 10; hidden = 64
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
model.add(LSTM(units=hidden, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/state/64/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/state/64/singlemae.npy",mae)
np.save("./pricedata/state/64/singlermse.npy",rmse)


##################################renewable pre state = 128####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 12; n_outputs = 10; hidden = 128
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
model.add(LSTM(units=hidden, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/state/128/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/state/128/singlemae.npy",mae)
np.save("./pricedata/state/128/singlermse.npy",rmse)

##################################renewable pre layer = 1####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 12; n_outputs = 10; hidden = 64
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
model.add(LSTM(units=hidden,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/layer/1/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/layer/1/singlemae.npy",mae)
np.save("./pricedata/layer/1/singlermse.npy",rmse)

##################################renewable pre layer = 2####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 12; n_outputs = 10; hidden = 64
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
model.add(LSTM(units=hidden, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/layer/2/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/layer/2/singlemae.npy",mae)
np.save("./pricedata/layer/2/singlermse.npy",rmse)


##################################renewable pre layer = 3####################################

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]

BACK_DAY = 12; n_outputs = 10; hidden = 64
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
model.add(LSTM(units=hidden, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=hidden))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test), verbose=2)
model.save("./pricedata/layer/3/point.h5")

#model = load_model("./pricedata/point.h5")

mae = [];rmse = []

for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
        
    test_predict = []; test_y = [];
    for i in range(BACK_DAY,BACK_DAY+600):
        test_copy = test.copy()
        test_predict_a = []
        for j in range(n_outputs):
            x_test = test_copy[i+j-BACK_DAY:i+j,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
            test_copy[i+j,0] = test_predict_v[0][0]
            test_predict_a.append(scaler_p.inverse_transform(test_predict_v)[0][0])
        test_predict.append(test_predict_a)
        test_y.append(scaler_p.inverse_transform([test[i:i+n_outputs,0]])[0])
    test_predict = np.array(test_predict); test_y = np.array(test_y)
    mae.append(mean_absolute_error(test_y, test_predict))
    rmse.append(np.sqrt(mean_squared_error(test_y, test_predict)))
    print('Test Mean Absolute Error:', mean_absolute_error(test_y, test_predict))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y, test_predict)))
    
mae,rmse = np.array(mae), np.array(rmse)    
np.save("./pricedata/layer/3/singlemae.npy",mae)
np.save("./pricedata/layer/3/singlermse.npy",rmse)