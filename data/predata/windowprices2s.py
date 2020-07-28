# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:27:25 2020

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
"""
##################################renewable pre Backday = 6####################################

BACK_DAY = 6;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=64, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/step/6/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/step/6/s2smae.npy",mae)
np.save("./pricedata/step/6/s2srmse.npy",rmse)

##################################renewable pre Backday = 12####################################

BACK_DAY = 12;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=64, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/step/12/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/step/12/s2smae.npy",mae)
np.save("./pricedata/step/12/s2srmse.npy",rmse)
"""
##################################renewable pre Backday = 24####################################

BACK_DAY = 24;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=64, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/step/24/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/step/24/s2smae.npy",mae)
np.save("./pricedata/step/24/s2srmse.npy",rmse)

##################################renewable pre state = 16####################################

BACK_DAY = 12;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=16, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/state/16/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/state/16/s2smae.npy",mae)
np.save("./pricedata/state/16/s2srmse.npy",rmse)

##################################renewable pre state = 64####################################

BACK_DAY = 12;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=64, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/state/64/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/state/64/s2smae.npy",mae)
np.save("./pricedata/state/64/s2srmse.npy",rmse)

##################################renewable pre state = 128####################################

BACK_DAY = 12;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=128, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/state/128/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/state/128/s2smae.npy",mae)
np.save("./pricedata/state/128/s2srmse.npy",rmse)

##################################renewable pre layer = 1####################################

BACK_DAY = 12;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=64, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=1, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/layer/1/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/layer/1/s2smae.npy",mae)
np.save("./pricedata/layer/1/s2srmse.npy",rmse)

##################################renewable pre layer = 2####################################

BACK_DAY = 12;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=64, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/layer/2/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/layer/2/s2smae.npy",mae)
np.save("./pricedata/layer/2/s2srmse.npy",rmse)

##################################renewable pre layer = 3####################################

BACK_DAY = 12;
dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
test_size = int(len(dataset_p) * 0.30)
train, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+test_size,:]
mae = [];rmse = []

for n_outputs in range(5):
#for n_outputs in range(1,11):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    model = Seq2Seq(output_dim=1, hidden_dim=64, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2, dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2)
    model.save_weights("./pricedata/layer/3/s2s"+str(n_outputs)+".h5")
    
    #model = Seq2Seq(output_dim=1, hidden_dim=10, output_length=n_outputs, input_shape=(BACK_DAY, train.shape[1]), peek=False, depth=2,dropout=0.2)
    #model.load_weights("./pricedata/s2s"+str(n_outputs)+".h5")
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_p.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_p.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./pricedata/layer/3/s2smae.npy",mae)
np.save("./pricedata/layer/3/s2srmse.npy",rmse)