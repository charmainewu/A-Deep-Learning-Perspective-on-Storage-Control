# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:06:22 2020

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
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
sns.set_style('white')
sns.despine()

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


###############################################################################################
k=34
x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, s2sonl[0,:k]/s2sofl[0,:k], linewidth=2,marker = "*", label='DETA')
axs.plot(x, s2slstm[0,:k]/s2sofl[0,:k], linewidth=2, label='S2S LSTM')
axs.plot(x, vectorlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, pointlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month11.pdf")

x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, s2sonl1[0,:k]/s2sofl1[0,:k], linewidth=2,marker = "*", label='DETA')
axs.plot(x, s2slstm1[0,:k]/s2sofl1[0,:k], linewidth=2, label='S2S LSTM')
axs.plot(x, vectorlstm1[0,:k]/s2sofl1[0,:k], linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, pointlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month12.pdf")

robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, robdeta, linewidth=2,marker = "*", label='DETA')
axs.plot(x, robs2s, linewidth=2, label='S2S LSTM')
axs.plot(x, robvector, linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, robpoint, linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month13.pdf")

#####################################################################################################
k=10
x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, s2sonl[0,:k]/s2sofl[0,:k], linewidth=2,marker = "*", label='DETA')
axs.plot(x, s2slstm[0,:k]/s2sofl[0,:k], linewidth=2, label='S2S LSTM')
axs.plot(x, vectorlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, pointlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month21.pdf")

x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, s2sonl1[0,:k]/s2sofl1[0,:k], linewidth=2,marker = "*", label='DETA')
axs.plot(x, s2slstm1[0,:k]/s2sofl1[0,:k], linewidth=2, label='S2S LSTM')
axs.plot(x, vectorlstm1[0,:k]/s2sofl1[0,:k], linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, pointlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month22.pdf")

robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, robdeta, linewidth=2,marker = "*", label='DETA')
axs.plot(x, robs2s, linewidth=2, label='S2S LSTM')
axs.plot(x, robvector, linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, robpoint, linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month23.pdf")

#############################################################################################

k=4
x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, s2sonl[0,:k]/s2sofl[0,:k], linewidth=2,marker = "*", label='DETA')
axs.plot(x, s2slstm[0,:k]/s2sofl[0,:k], linewidth=2, label='S2S LSTM')
axs.plot(x, vectorlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, pointlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month31.pdf")

x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, s2sonl1[0,:k]/s2sofl1[0,:k], linewidth=2,marker = "*", label='DETA')
axs.plot(x, s2slstm1[0,:k]/s2sofl1[0,:k], linewidth=2, label='S2S LSTM')
axs.plot(x, vectorlstm1[0,:k]/s2sofl1[0,:k], linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, pointlstm[0,:k]/s2sofl[0,:k], linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month32.pdf")

robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

x = list(range(1,k+1))
fig, axs = plt.subplots()
axs.plot(x, robdeta, linewidth=2,marker = "*", label='DETA')
axs.plot(x, robs2s, linewidth=2, label='S2S LSTM')
axs.plot(x, robvector, linewidth=2, dashes = [6,1], label='Vector LSTM')
axs.plot(x, robpoint, linewidth=2, dashes = [1,1], label='Point LSTM')

#axs.grid(True, linestyle='-.')
plt.xlabel('Month', fontsize = 17)
plt.ylabel(r'$ \alpha $', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs.legend()
plt.savefig("./figure/month33.pdf")

##################################################################################


    
    




