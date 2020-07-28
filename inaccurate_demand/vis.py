# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:47:49 2020

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
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt

###############################################################################
deta = np.load("./data/50/cost_deta.npy")
ofl = np.load("./data/50/cost_ofl.npy")
nos = np.load("./data/50/cost_nos.npy")
mpc = np.load("./data/50/cost_mpc.npy")
thb = np.load("./data/50/cost_thb.npy")
rl = np.load("./data/50/cost_rl.npy")
s2slstm = np.load("./data/50/cost_seta.npy")
vectorlstm = np.load("./data/50/cost_veta.npy")
pointlstm = np.load("./data/50/cost_peta.npy")

x = list(range(100,3600,300))
    
fig, axs = plt.subplots(figsize=(6, 4),constrained_layout=True)
#axs.plot(x, nos[0:36:3]/ofl[0:36:3],  linewidth=2,c = 'royalblue',marker = "o", markersize = 10, label='No Storage')
axs.plot(x, rl[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "p",  markersize = 12,c='lightseagreen',label='RL')
axs.plot(x, mpc[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "h",  markersize = 12,c='gold',label='MPC')
axs.plot(x, thb[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "<",  markersize = 12,c='royalblue',label='THB')
axs.plot(x, deta[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "*",  markersize = 12,c='darkgray',label='DETA')
axs.plot(x, s2slstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = ">", markersize = 10,c='orange',label='SPTA')
axs.plot(x, vectorlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "s", markersize = 10,c='salmon', label='VPTA')
axs.plot(x, pointlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "v", markersize = 10,c='yellowgreen', label='PPTA')


plt.xlabel('Time/h', fontsize = 20)
plt.ylabel(r'$ \alpha $', fontsize = 20)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
axs.legend(fontsize = 14,loc=1)
plt.savefig("./figure/inaccur1.pdf")


###############################################################################
deta = np.load("./data/100/cost_deta.npy")
ofl = np.load("./data/100/cost_ofl.npy")
nos = np.load("./data/100/cost_nos.npy")
mpc = np.load("./data/100/cost_mpc.npy")
thb = np.load("./data/100/cost_thb.npy")
rl = np.load("./data/100/cost_rl.npy")
s2slstm = np.load("./data/100/cost_seta.npy")
vectorlstm = np.load("./data/100/cost_veta.npy")
pointlstm = np.load("./data/100/cost_peta.npy")

x = list(range(100,3600,300))
    
fig, axs = plt.subplots(figsize=(6, 4),constrained_layout=True)
#axs.plot(x, nos[0:36:3]/ofl[0:36:3],  linewidth=2,c = 'royalblue',marker = "o", markersize = 10, label='No Storage')
axs.plot(x, rl[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "p",  markersize = 12,c='lightseagreen',label='RL')
axs.plot(x, mpc[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "h",  markersize = 12,c='gold',label='MPC')
axs.plot(x, thb[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "<",  markersize = 12,c='royalblue',label='THB')
axs.plot(x, deta[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "*",  markersize = 12,c='darkgray',label='DETA')
axs.plot(x, s2slstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = ">", markersize = 10,c='orange',label='SPTA')
axs.plot(x, vectorlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "s", markersize = 10,c='salmon', label='VPTA')
axs.plot(x, pointlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "v", markersize = 10,c='yellowgreen', label='PPTA')

plt.xlabel('Time/h', fontsize = 20)
plt.ylabel(r'$ \alpha $', fontsize = 20)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
#axs.legend(fontsize = 14,loc=3)
plt.savefig("./figure/inaccur2.pdf")

###############################################################################
deta = np.load("./data/500/cost_deta.npy")
ofl = np.load("./data/500/cost_ofl.npy")
nos = np.load("./data/500/cost_nos.npy")
mpc = np.load("./data/500/cost_mpc.npy")
thb = np.load("./data/500/cost_thb.npy")
rl = np.load("./data/500/cost_rl.npy")
s2slstm = np.load("./data/500/cost_seta.npy")
vectorlstm = np.load("./data/500/cost_veta.npy")
pointlstm = np.load("./data/500/cost_peta.npy")

x = list(range(100,3600,300))
    
fig, axs = plt.subplots(figsize=(6, 4),constrained_layout=True)
#axs.plot(x, nos[0:36:3]/ofl[0:36:3],  linewidth=2,c = 'royalblue',marker = "o", markersize = 10, label='No Storage')
axs.plot(x, rl[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "p",  markersize = 12,c='lightseagreen',label='RL')
axs.plot(x, mpc[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "h",  markersize = 12,c='gold',label='MPC')
axs.plot(x, thb[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "<",  markersize = 12,c='royalblue',label='THB')
axs.plot(x, deta[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "*",  markersize = 12,c='darkgray',label='DETA')
axs.plot(x, s2slstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = ">", markersize = 10,c='orange',label='SPTA')
axs.plot(x, vectorlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "s", markersize = 10,c='salmon', label='VPTA')
axs.plot(x, pointlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "v", markersize = 10,c='yellowgreen', label='PPTA')

plt.xlabel('Time/h', fontsize = 20)
plt.ylabel(r'$ \alpha $', fontsize = 20)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
#axs.legend()
plt.savefig("./figure/inaccur3.pdf")
