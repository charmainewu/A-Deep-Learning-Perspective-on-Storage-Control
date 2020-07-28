# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:24:48 2020

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

import seaborn as sns
import matplotlib as mpl
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt


s2sonl = np.load("./figure/s2sonl.npy")
s2sofl = np.load("./figure/s2sofl.npy")
s2slstm = np.load("./figure/s2slstm.npy")
s2sonl1 = np.load("./figure/s2sonl1.npy")
s2sofl1 = np.load("./figure/s2sofl1.npy")
s2slstm1 = np.load("./figure/s2slstm1.npy")
vectoronl = np.load("./figure/vectoronl.npy")
vectorofl = np.load("./figure/vectorofl.npy")
vectorlstm = np.load("./figure/vectorlstm.npy")
vectoronl1 = np.load("./figure/vectoronl1.npy")
vectorofl1 = np.load("./figure/vectorofl1.npy")
vectorlstm1 = np.load("./figure/vectorlstm1.npy")
pointonl = np.load("./figure/pointonl.npy")
pointofl = np.load("./figure/pointofl.npy")
pointlstm = np.load("./figure/pointlstm.npy")
pointonl1 = np.load("./figure/pointonl1.npy")
pointofl1 = np.load("./figure/pointofl1.npy")
pointlstm1 = np.load("./figure/pointlstm1.npy")

###############################################################################

labels = ['DETA','VPTA','SPTA','PPTA']
fs = 23

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 10), sharey=False)

k=35
onl = s2sonl[0,:k]/s2sofl[0,:k]
vector = vectorlstm[0,:k]/s2sofl[0,:k]
s2s = s2slstm[0,:k]/s2sofl[0,:k]
point = s2slstm[0,:k]/s2sofl[0,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[0,0].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[0,0].set_title('# Month = 1, Testing', fontsize=fs)

k=10
onl = s2sonl[1,:k]/s2sofl[1,:k]
vector = vectorlstm[1,:k]/s2sofl[1,:k]
s2s = s2slstm[1,:k]/s2sofl[1,:k]
point = s2slstm[1,:k]/s2sofl[1,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[0,1].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[0,1].set_title('# Month = 3, Testing', fontsize=fs)

k=4
onl = s2sonl[2,:k]/s2sofl[2,:k]
vector = vectorlstm[2,:k]/s2sofl[2,:k]
s2s = s2slstm[2,:k]/s2sofl[2,:k]
point = s2slstm[2,:k]/s2sofl[2,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[0,2].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[0,2].set_title('# Month = 6, Testing', fontsize=fs)

###############################################################################
k=35
onl = s2sonl1[0,:k]/s2sofl1[0,:k]
vector = vectorlstm1[0,:k]/s2sofl1[0,:k]
s2s = s2slstm1[0,:k]/s2sofl1[0,:k]
point = s2slstm1[0,:k]/s2sofl1[0,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[1,0].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[1,0].set_title('# Month = 1, Training', fontsize=fs)

k=10
onl = s2sonl1[1,:k]/s2sofl1[1,:k]
vector = vectorlstm1[1,:k]/s2sofl1[1,:k]
s2s = s2slstm1[1,:k]/s2sofl1[1,:k]
point = s2slstm1[1,:k]/s2sofl1[1,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[1,1].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[1,1].set_title('# Month = 3, Training', fontsize=fs)

k=4
onl = s2sonl1[2,:k]/s2sofl1[2,:k]
vector = vectorlstm1[2,:k]/s2sofl1[2,:k]
s2s = s2slstm1[2,:k]/s2sofl1[2,:k]
point = s2slstm1[2,:k]/s2sofl1[2,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[1,2].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[1,2].set_title('# Month = 6, Training', fontsize=fs)

###############################################################################
k=35

onl = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
s2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
vector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
point = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[2,0].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[2,0].set_title('# Month = 1, Robustness', fontsize=fs)

k=10
onl = s2sonl1[1,:k]/s2sofl1[1,:k]-s2sonl[1,:k]/s2sofl[1,:k]
s2s = s2slstm1[1,:k]/s2sofl1[1,:k]-s2slstm[1,:k]/s2sofl[1,:k]
vector =vectorlstm1[1,:k]/s2sofl1[1,:k]-vectorlstm[1,:k]/s2sofl[1,:k]
point = pointlstm1[1,:k]/pointofl1[1,:k]-pointlstm[1,:k]/s2sofl[1,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[2,1].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[2,1].set_title('# Month = 3, Robustness', fontsize=fs)

k=4
onl = s2sonl1[2,:k]/s2sofl1[2,:k]-s2sonl[2,:k]/s2sofl[2,:k]
s2s = s2slstm1[2,:k]/s2sofl1[2,:k]-s2slstm[2,:k]/s2sofl[2,:k]
vector =vectorlstm1[2,:k]/s2sofl1[2,:k]-vectorlstm[2,:k]/s2sofl[2,:k]
point = pointlstm1[2,:k]/pointofl1[2,:k]-pointlstm[2,:k]/s2sofl[2,:k]

data = []
data.append(onl)
data.append(vector)
data.append(s2s)
data.append(point)
data = np.array(data)
data = np.transpose(data)

axes[2,2].boxplot(data, labels=labels, showmeans=True, meanline=True)
axes[2,2].set_title('# Month = 6, Robustness', fontsize=fs)   
    

fs2 = 19
axes[0,0].tick_params(labelsize=fs2)
axes[0,1].tick_params(labelsize=fs2) 
axes[0,2].tick_params(labelsize=fs2)

axes[1,0].tick_params(labelsize=fs2) 
axes[1,1].tick_params(labelsize=fs2) 
axes[1,2].tick_params(labelsize=fs2)  

axes[2,0].tick_params(labelsize=fs2)
axes[2,1].tick_params(labelsize=fs2)
axes[2,2].tick_params(labelsize=fs2)
    
fig.tight_layout()
plt.savefig("./figure/rob1.pdf")

