# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:37:50 2020

@author: jmwu
"""


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
import seaborn as sns
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

###############################################################################
def isDecompose(at,B,x0):
    Acc_sum = 0
    Acc = np.zeros(len(at))
    for i in range(len(at)):
        Acc[i] = Acc_sum + at[i]; Acc_sum = Acc_sum + at[i]
    AccB = Acc + B
    sadwAB = np.zeros(int(max(Acc)+1))
    for i in range(len(at)):
        if AccB[i] <= max(Acc):
            sadwAB[int(AccB[i])] = 1
        sadwAB[int(Acc[i])] = 1
    sadwA = np.zeros(int(max(Acc)+1))
    sadwB = np.zeros(int(max(Acc)+1))
    i = 0; j = int(Acc[0]); 
    while(j <= max(Acc) and i+1<=len(at)-1):
        while(i+1<=len(at)-1 and Acc[i+1]==Acc[i]):
            try:
                sadwA[j] = i+1;i = i + 1
            except:
                break
        while(i+1<=len(at)-1 and Acc[i+1]>Acc[i]):
            k = Acc[i+1]-Acc[i]
            try:
                sadwA[j] = i + 1
            except: 
                break
            while(k > 0):
                j = j + 1
                try:
                    sadwA[j] = i + 1;k = k - 1
                except:
                    break
            i = i + 1
    i = 0; j = int(AccB[0]); 
    while(j <= max(Acc) and i+1<=len(at)-1):
        while(i+1<=len(at)-1 and AccB[i+1]==AccB[i]):
            i = i + 1
        while(i+1<=len(at)-1 and AccB[i+1]>AccB[i]):
            k = AccB[i+1]-AccB[i]
            while(k > 0):
                j = j + 1
                try:
                    sadwB[j] = i + 1
                    k = k - 1
                except:
                    break
            i = i + 1
    a_index =np.where(sadwAB==1)[0]
    a = [];ts = [];tnz = [];
    for i in range(len(a_index)-1):
        a.append(a_index[i+1]-a_index[i])
        ts.append(sadwB[a_index[i+1]])
        tnz.append(sadwA[a_index[i]])
    Trunc_sum = a[0]; t = 0; del_list = [];
    while(Trunc_sum <= x0):
        del_list.append(t)
        t = t + 1
        try:
            Trunc_sum = Trunc_sum + a[t]
        except:
            break
    for i in del_list:
        del a[i]
        del ts[i]
        del tnz[i]
    a[0] = Trunc_sum - x0
    return a, ts, tnz

def Athb(theta,B,t,at,pt,xt0,tnz):
    mu_c = 1000000; mu_d =1000000
    if t == tnz:
        dt =  min(at,mu_d,xt0)
        vat = at - dt
        vbt = 0
        xt = xt0+vbt-dt
        return xt,dt,vat,vbt
    if pt<=theta:
        dt = 0
        vat = at
        vbt = min(max(B-xt0,0),mu_c)
    else:
        dt =  min(at,mu_d,xt0)
        vat = at - dt
        vbt = 0
    xt = xt0+vbt-dt
    return xt,dt,vat,vbt
            
def Athb_ld(theta,ts,tnz,abar,pt,t,xt0):
    if t == tnz:
        abart = abar
    else:
        abart = 0
    return Athb(theta,abar,t,abart,pt,xt0,tnz)

def Aofl(ts,tnz,abar,p):
    mu_c = 1000000; mu_d =1000000
    a = np.zeros(len(p))
    a[int(tnz)] = abar
    xt = np.zeros(len(p))
    dt = np.zeros(len(p))
    vat = np.zeros(len(p))
    vbt = np.zeros(len(p))
    try:
        p_min = min(p[int(ts):int(tnz+1)])
    except:
        p_min = 1e8
            
    if p[int(ts)]==p_min:
        dt[int(ts)] = 0
        vat[int(ts)] = a[int(ts)]
        vbt[int(ts)] = min(max(abar-0,0),mu_c)
        xt[int(ts)] = 0 + vbt[int(ts)] - dt[int(ts)]
    else:
        dt[int(ts)] = min(a[int(ts)],mu_d,0)
        vat[int(ts)] = a[int(ts)]-dt[int(ts)]
        vbt[int(ts)] = 0
        xt[int(ts)] = 0 + vbt[int(ts)] - dt[int(ts)]
    
    for t in range(int(ts+1),int(tnz)):
        if p[t] == p_min:
            dt[t] = 0
            vat[t] = a[t]
            vbt[t] = min(max(abar-xt[t-1],0),mu_c)
            xt[t] = xt[t-1] + vbt[t] - dt[t]
        else:
            dt[t] = min(a[t],mu_d,xt[t-1])
            vat[t] = a[t]-dt[t]
            vbt[t] = 0
            xt[t] = xt[t-1] + vbt[t] - dt[t]
            
    dt[int(tnz)] = min(a[int(tnz)],mu_d,xt[int(tnz-1)])
    vat[int(tnz)] = a[int(tnz)]-dt[int(tnz)]
    vbt[int(tnz)] = 0
    xt[int(tnz)] = xt[int(tnz-1)] + vbt[int(tnz)] - dt[int(tnz)]
    
    return xt,dt,vat,vbt

def Aofl_hat(ao,po,B):
    a = ao.copy() ; p = po.copy()
    x = np.zeros(int(len(a)))
    d = np.zeros(int(len(a)))
    va = np.zeros(int(len(a)))
    vb = np.zeros(int(len(a)))
    x0 = 0; x[0] = 0
    abar, ts, tnz = isDecompose(a,B,x0)
    cost_shot = np.zeros(len(abar))
    for i in range(len(abar)):
        xt,dt,vat,vbt = Aofl(ts[i],tnz[i],abar[i],p)
        for t in range(int(ts[i]),int(tnz[i]+1)):
            x[t] = x[t] + xt[t]
            d[t] = d[t] + dt[t]
            va[t] = va[t] + vat[t]
            vb[t] = vb[t] + vbt[t]
            a[t] = a[t] - dt[t] - vat[t]
            if vat[t]+vbt[t]>0:
                cost_shot[i] = cost_shot[i] + p[t]*(vat[t]+vbt[t])
    return x,d,va,vb,cost_shot

def estimate_gmm(X):
    X = X.reshape(-1,1)
    bic = []; lowest_bic = np.infty;
    n_components_range = range(1, 7)
    cv_types = ['spherical']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm    
    clf = best_gmm
    return clf
    
def Norm_pdfx(x):
    return x

def Atheta_gmm(clf,T):
    k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
    t = int(T-2); theta = np.zeros(int(T))
    trunc = 0; re = 0;
    for i in range(k):
        trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
        re = re + por[i]*norm.expect(Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
    theta[int(T-2)] = re/trunc
    while(t>0):
        t = t-1; re1 = 0; re2 = 0; trunc = 0
        for i in range(k):
            trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
            re1 = re1 + por[i]*norm.expect(Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0,ub= theta[t+1])
            re2 = re2 + por[i]*theta[t+1] * (1-norm.cdf(theta[t+1],loc = mean[i], scale = np.sqrt(std[i])))
        theta[t] = (re1+re2)/trunc
    return theta

def Aour_hat_gmm(ao,po,B,clf,x0):
    a = ao.copy(); p = po.copy()
    x = np.zeros(int(len(a)))
    d = np.zeros(int(len(a)))
    va = np.zeros(int(len(a)))
    vb = np.zeros(int(len(a)))

    abar, ts, tnz = isDecompose(a,B,x0)
    xi = np.zeros((len(abar),len(a)))
    theta = np.zeros((len(abar),len(a)))
    cost_shot = np.zeros((len(abar),len(a)))
    
    Theta = Atheta_gmm(clf,len(a))
    
    for t in range(len(a)):
        for i in range(len(abar)):
            if t>=ts[i] and t<=tnz[i]:
                theta = Theta[len(a)-int(tnz[i]-t)-1]
                if t==ts[i]:
                    xt,dt,vat,vbt = Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
                                            t,0)
                    x[t] = x[t] + xt
                    d[t] = d[t] + dt
                    va[t] = va[t] + vat
                    vb[t] = vb[t] + vbt
                    a[t] = a[t] - dt - vat 
                    xi[i,t] = xt
                    if vat+vbt>0:
                        cost_shot[i,t] =  p[t]*(vat+vbt)
                else:
                    xt,dt,vat,vbt = Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
                                            t,xi[i,t-1])
                    x[t] = x[t] + xt
                    d[t] = d[t] + dt
                    va[t] = va[t] + vat
                    vb[t] = vb[t] + vbt
                    a[t] = a[t] - dt - vat 
                    xi[i,t] = xt
                    if vat+vbt>0:
                        cost_shot[i,t] =  p[t]*(vat+vbt)
    return x,d,va,vb,cost_shot

def Aour_demand(d_s,c_s,a,p,B):
    T = len(p)
    x = np.zeros(T)
    d = np.zeros(T)
    va = np.zeros(T)
    vb = np.zeros(T)
    cost = 0
    for t in range(T):
        if t == 0 :
            d[t] = min(a[t],0,d_s[t])
            va[t] = a[t] - d[t]
            vb[t] = min(B-0+d[t],c_s[t])
            x[t] = 0 +vb[t]-d[t]
            cost = cost + (va[t]+vb[t])*p[t]
        else:
            d[t] = min(a[t],x[t-1],d_s[t])
            va[t] = a[t] - d[t]
            vb[t] = min(B-x[t-1]+d[t],c_s[t])
            x[t] = x[t-1] +vb[t]-d[t]
            cost = cost + (va[t]+vb[t])*p[t]
    return cost,x[t]
###############################################################################

def pred_step(data,scaler,model,t,TS,TD):

    TOTAL_LEN = len(data[:,0])
    VALID_SAT = len(data[N_TRAIN+TS+t:,0])
    VALID_END = len(data[N_TRAIN+TD:,0])
    
    inputs = data[TOTAL_LEN - VALID_SAT - BACK_DAY:
        TOTAL_LEN - VALID_END,:]
    test_copy = inputs.copy()
    
    pred = []
    for i in range(BACK_DAY,inputs.shape[0]):
        if (i-BACK_DAY)%10==0:
            x_test = test_copy[i-BACK_DAY:i,:]
            x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
            test_predict_v = model.predict(x_test)
           
            for j in range(10):
                #print(test_predict_v[0][j][0])
                if test_predict_v[0][j][0]>1:
                    test_copy[i,0] = 1
                    pred.append(1)
                elif test_predict_v[0][j][0]<0:
                    test_copy[i,0] = 0
                    pred.append(0)
                else:
                    test_copy[i,0] = test_predict_v[0][j][0]
                    pred.append(test_predict_v[0][j][0])
    pred = np.array(pred)
    
    test_predict = scaler.inverse_transform(pred.reshape(-1, 1))

    return test_predict

def Aour_pred_transfer(data_s,data,scaler,abar,ts,tnz,model,TS,TD):
    #data = scaler.inverse_transform(scaled_data)
    test_price = data[N_TRAIN+TS-1:N_TRAIN+TD].values
    buy = np.zeros(len(abar))
    cost = np.zeros(len(abar))
    discharge = np.zeros(len(test_price))
    charge = np.zeros(len(test_price))
    
    for t in range(len(test_price)):
        try:
            prediction = pred_step(data_s,scaler,model,t,TS,TD)
        except:
            prediction = test_price[t]

        for i in range(len(abar)):
            #print(ts[i],tnz[i],t,prediction[:int(tnz[i])-t].values)
            if t==tnz[i] and buy[i] != 1:
                cost[i] = abar[i]*test_price[t]
                discharge[t] = discharge[t] + 0
                charge[t] = charge[t] + 0
                buy[i] = 1
                continue
            elif t==tnz[i] and buy[i] == 1:
                discharge[t] = discharge[t] + abar[i]
                charge[t] = charge[t] + 0
                buy[i] = 1
                continue
            if t>=ts[i] and t<tnz[i] and buy[i] != 1:
                #print(ts[i],tnz[i],t,test_price[t],test_price[t+1],prediction[:int(tnz[i])-t])
                if test_price[t] <= min(prediction[:int(tnz[i])-t]):
                    cost[i] = abar[i]*test_price[t]
                    discharge[t] = discharge[t] + 0
                    charge[t] = charge[t] + abar[i]
                    buy[i] = 1
    return discharge,charge,sum(cost)

def Aour_hat_gmm_rew(a,p,r,B,clf,W,model_pv,r_data,scaler_r,data_sr):
    n_interval = int(len(a)/W); xc = 0;
    cost = np.zeros(n_interval)
    for n in range(n_interval):
        r_temp = pred_step(data_sr, scaler_r, model_pv, 0,(n)*W,(n+1)*W)
        #scaled_data,scaler,model,t,TS,TD
        r_temp = r_temp.astype(int)
        r_temp = r_temp.reshape(len(r_temp),1)
   
        a_temp = a[(n)*W:(n+1)*W]
        p_temp = p[(n)*W:(n+1)*W]
        r_real = r[(n)*W:(n+1)*W]
        
        d_temp = (a_temp - r_temp)
        d_real = (a_temp - r_real)
        
        for i in range(len(d_temp)):
            d_temp[i] = max(d_temp[i],0)
        
        d_temp = d_temp.astype(int); d_real = d_real.astype(int) 
        d_temp = np.r_[np.array([[0]]),d_temp]; d_real = np.r_[np.array([[0]]),d_real]
        p_temp = np.r_[np.array([p[(n)*W-1]]),p_temp]
        
        x,d,va,vb,cost_shot = Aour_hat_gmm(d_temp,p_temp,B,clf,xc)
        cost[n],xc = Aour_demand(d,vb,d_real,p_temp,B)

    return sum(cost)

def Aour_pred_transfer_rew(a,p,r,B,W,model_price,model_pv,p_data,r_data,scaler_p,scaler_r,data_sr,data_sp):
    n_interval = int(len(a)/W); xc = 0;
    cost = np.zeros(n_interval)
    for n in range(n_interval):
        r_temp = pred_step(data_sr,scaler_r,model_pv,0,(n)*W,(n+1)*W)
        #scaled_data,scaler,model,t,TS,TD
        r_temp = r_temp.astype(int)
        r_temp = r_temp.reshape(len(r_temp),1)
        
        a_temp = a[(n)*W:(n+1)*W]
        p_temp = p[(n)*W:(n+1)*W]
        r_real = r[(n)*W:(n+1)*W]

        d_temp = (a_temp - r_temp)
        d_real = (a_temp - r_real)
        
        for i in range(len(d_temp)):
            d_temp[i] = max(d_temp[i],0)
        
        d_temp = d_temp.astype(int); d_real = d_real.astype(int) 
        d_temp = np.r_[np.array([[0]]),d_temp]; d_real = np.r_[np.array([[0]]),d_real]
        p_temp = np.r_[np.array([p[(n)*W-1]]),p_temp]
        
        abar, ts, tnz = isDecompose(d_temp,B,xc)
        d,vb,cost_ori = Aour_pred_transfer(data_sp,p_data,scaler_p,abar,ts,tnz,model_price,(n)*W,(n+1)*W)
        cost[n],xc = Aour_demand(d,vb,d_real,p_temp,B)

    return sum(cost)

if __name__ == "__main__":
    
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

    dataset_r  = np.hstack((r,w,h))
    dataset_p  = np.hstack((p,w,h))
    ###########################################################################
    
    l = load_data; p = price_data; r = renewable_data
    K = 36; W = 100; B = 1000; BACK_DAY = 24;n_outputs=10
    
    costlstm = np.zeros((4,K))
    costonl = np.zeros((4,K))
    costofl = np.zeros((4,K))
    
    costlstm1 = np.zeros((4,K))
    costonl1 = np.zeros((4,K))
    costofl1 = np.zeros((4,K))
    
    count = 0
    for n_month in [1,3,6,12]:
        for k in range(1,36):
            print(n_month,k)
            try:
                model_pv = load_model("./winddata/vectorr"+str(k)+str(n_month)+".h5")
                model_price = load_model("./pricedata/vectorp"+str(k)+str(n_month)+".h5")
                
                N_TRAIN = k*int(len(dataset_p) *n_month* 0.027)
                train_size = int(len(dataset_p) *n_month* 0.027)
                N_TEST = int(len(dataset_p) *n_month* 0.027)
                N_TEST = int(N_TEST/W)*W
                ###########################################################################
                l_test = l.values[N_TRAIN:N_TRAIN+N_TEST,:]
                r_test = r.values[N_TRAIN:N_TRAIN+N_TEST,:]
                p_test = p.values[N_TRAIN:N_TRAIN+N_TEST,:]
                ########################True###############################################
                n_test = (l_test - r_test)
                n_test = np.r_[np.array([[0]]),n_test]
                n_test = n_test.astype(int)
                v_test = np.r_[np.array([p.values[N_TRAIN-1,:]]),p_test]
                x,d,va,vb,costo = Aofl_hat(n_test,v_test,B)
                costofl[count,k-1] = sum(costo)
                print(costofl[count,k-1])
                ########################Scheduling#########################################
                clf = estimate_gmm(p.values[N_TRAIN:N_TRAIN+train_size,:])
                costonl[count,k-1] = Aour_hat_gmm_rew(l_test,p_test,r_test,B,clf,W,model_pv,r,scaler_r,dataset_r)
                costlstm[count,k-1] = Aour_pred_transfer_rew(l_test,p_test,r_test,B,W,model_price,model_pv,p,r,scaler_p,scaler_r,dataset_r,dataset_p)
                print(costonl[count,k-1])
                print(costlstm[count,k-1])
                
                N_TRAIN = k*int(len(dataset_p) *n_month* 0.027)+int(len(dataset_p) *n_month* 0.027)
                train_size = int(len(dataset_p) *n_month* 0.027)
                N_TEST = int(len(dataset_p) *n_month* 0.027)
                N_TEST = int(N_TEST/W)*W
                ###########################################################################        
                l_test = l.values[N_TRAIN:N_TRAIN+N_TEST,:]
                r_test = r.values[N_TRAIN:N_TRAIN+N_TEST,:]
                p_test = p.values[N_TRAIN:N_TRAIN+N_TEST,:]
                ########################True###############################################
                n_test = (l_test - r_test)
                n_test = np.r_[np.array([[0]]),n_test]
                n_test = n_test.astype(int)
                v_test = np.r_[np.array([p.values[N_TRAIN-1,:]]),p_test]
                x,d,va,vb,costo = Aofl_hat(n_test,v_test,B)
                costofl1[count,k-1] = sum(costo)
                print(costofl1[count,k-1])
                ########################Scheduling#########################################
                clf = estimate_gmm(p.values[N_TRAIN-train_size:N_TRAIN,:])
                costonl1[count,k-1] = Aour_hat_gmm_rew(l_test,p_test,r_test,B,clf,W,model_pv,r,scaler_r,dataset_r)
                costlstm1[count,k-1] = Aour_pred_transfer_rew(l_test,p_test,r_test,B,W,model_price,model_pv,p,r,scaler_p,scaler_r,dataset_r,dataset_p)
                print(costonl1[count,k-1])
                print(costlstm1[count,k-1])
            except:
                continue
        count = count +1
            
    roblstm = abs(costlstm/costofl-costlstm1/costofl1)
    robonl = abs(costonl/costofl-costonl1/costofl1)
    
    np.save("./figure/vectorlstm.npy",costlstm)
    np.save("./figure/vectoronl.npy",costonl)
    np.save("./figure/vectorofl.npy",costofl)
    np.save("./figure/vectorlstm1.npy",costlstm1)
    np.save("./figure/vectoronl1.npy",costonl1)
    np.save("./figure/vectorofl1.npy",costofl1)
        

        