# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:22:35 2020

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

#sns.set_style('white')
#sns.despine()

fs= 16
fs1 = 23
fs2= 19
###############################################################################
rs2smae16 = np.load("./winddata/state/16/s2smae.npy")/83000
rs2srmse16 = np.load("./winddata/state/16/s2srmse.npy")/83000
rvectormae16 = np.load("./winddata/state/16/vectormae.npy")/83000
rvectorrmse16 = np.load("./winddata/state/16/vectorrmse.npy")/83000
rpointmae16 = np.load("./winddata/state/16/singlemae.npy")/83000
rpointrmse16 = np.load("./winddata/state/16/singlermse.npy")/83000
ps2smae16 = np.load("./pricedata/state/16/s2smae.npy")/83000
ps2srmse16 = np.load("./pricedata/state/16/s2srmse.npy")/83000
pvectormae16 = np.load("./pricedata/state/16/vectormae.npy")/83000
pvectorrmse16 = np.load("./pricedata/state/16/vectorrmse.npy")/83000
ppointmae16 = np.load("./pricedata/state/16/singlemae.npy")/83000
ppointrmse16 = np.load("./pricedata/state/16/singlermse.npy")/83000
###############################################################################
rs2smae64 = np.load("./winddata/state/64/s2smae.npy")/83000
rs2srmse64 = np.load("./winddata/state/64/s2srmse.npy")/83000
rvectormae64 = np.load("./winddata/state/64/vectormae.npy")/83000
rvectorrmse64 = np.load("./winddata/state/64/vectorrmse.npy")/83000
rpointmae64 = np.load("./winddata/state/64/singlemae.npy")/83000
rpointrmse64 = np.load("./winddata/state/64/singlermse.npy")/83000
ps2smae64 = np.load("./pricedata/state/64/s2smae.npy")/83000
ps2srmse64 = np.load("./pricedata/state/64/s2srmse.npy")/83000
pvectormae64 = np.load("./pricedata/state/64/vectormae.npy")/83000
pvectorrmse64 = np.load("./pricedata/state/64/vectorrmse.npy")/83000
ppointmae64 = np.load("./pricedata/state/64/singlemae.npy")/83000
ppointrmse64 = np.load("./pricedata/state/64/singlermse.npy")/83000
###############################################################################
rs2smae128 = np.load("./winddata/state/128/s2smae.npy")/83000
rs2srmse128 = np.load("./winddata/state/128/s2srmse.npy")/83000
rvectormae128 = np.load("./winddata/state/128/vectormae.npy")/83000
rvectorrmse128 = np.load("./winddata/state/128/vectorrmse.npy")/83000
rpointmae128 = np.load("./winddata/state/128/singlemae.npy")/83000
rpointrmse128 = np.load("./winddata/state/128/singlermse.npy")/83000
ps2smae128 = np.load("./pricedata/state/128/s2smae.npy")/83000
ps2srmse128 = np.load("./pricedata/state/128/s2srmse.npy")/83000
pvectormae128 = np.load("./pricedata/state/128/vectormae.npy")/83000
pvectorrmse128 = np.load("./pricedata/state/128/vectorrmse.npy")/83000
ppointmae128 = np.load("./pricedata/state/128/singlemae.npy")/83000
ppointrmse128 = np.load("./pricedata/state/128/singlermse.npy")/83000
###############################################################################

x = [1,10,20,30,40]
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 9), sharey=True, constrained_layout=True)
axs[0,0].plot(x, rs2smae16, linewidth=2,marker = "*",markersize = 13, c = 'royalblue',label='# hidden state = 16')
axs[0,0].plot(x, rs2smae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# hidden state = 64')
axs[0,0].plot(x, rs2smae128, linewidth=2, marker = "^", markersize = 13,c = 'orange',label='# hidden state = 128')


axs[0,1].plot(x, rvectormae16, linewidth=2,marker = "*",markersize = 13, c = 'royalblue',label='# hidden state = 16')
axs[0,1].plot(x, rvectormae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# hidden state = 64')
axs[0,1].plot(x, rvectormae128, linewidth=2, marker = "^",markersize = 13, c = 'orange',label='# hidden state = 128')


axs[0,2].plot(x, rpointmae16, linewidth=2,marker = "*", markersize = 13,c = 'royalblue',label='# hidden state = 16')
axs[0,2].plot(x, rpointmae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# hidden state = 64')
axs[0,2].plot(x, rpointmae128, linewidth=2, marker = "^",markersize = 13,c = 'orange', label='# hidden state = 128')


###############################################################################
rs2smae16 = np.load("./winddata/step/6/s2smae.npy")/83000
rs2srmse16 = np.load("./winddata/step/6/s2srmse.npy")/83000
rvectormae16 = np.load("./winddata/step/6/vectormae.npy")/83000
rvectorrmse16 = np.load("./winddata/step/6/vectorrmse.npy")/83000
rpointmae16 = np.load("./winddata/step/6/singlemae.npy")/83000
rpointrmse16 = np.load("./winddata/step/6/singlermse.npy")/83000
ps2smae16 = np.load("./pricedata/step/6/s2smae.npy")/83000
ps2srmse16 = np.load("./pricedata/step/6/s2srmse.npy")/83000
pvectormae16 = np.load("./pricedata/step/6/vectormae.npy")/83000
pvectorrmse16 = np.load("./pricedata/step/6/vectorrmse.npy")/83000
ppointmae16 = np.load("./pricedata/step/6/singlemae.npy")/83000
ppointrmse16 = np.load("./pricedata/step/6/singlermse.npy")/83000
###############################################################################
rs2smae64 = np.load("./winddata/step/12/s2smae.npy")/83000
rs2srmse64 = np.load("./winddata/step/12/s2srmse.npy")/83000
rvectormae64 = np.load("./winddata/step/12/vectormae.npy")/83000
rvectorrmse64 = np.load("./winddata/step/12/vectorrmse.npy")/83000
rpointmae64 = np.load("./winddata/step/12/singlemae.npy")/83000
rpointrmse64 = np.load("./winddata/step/12/singlermse.npy")/83000
ps2smae64 = np.load("./pricedata/step/12/s2smae.npy")/83000
ps2srmse64 = np.load("./pricedata/step/12/s2srmse.npy")/83000
pvectormae64 = np.load("./pricedata/step/12/vectormae.npy")/83000
pvectorrmse64 = np.load("./pricedata/step/12/vectorrmse.npy")/83000
ppointmae64 = np.load("./pricedata/step/12/singlemae.npy")/83000
ppointrmse64 = np.load("./pricedata/step/12/singlermse.npy")/83000
###############################################################################
rs2smae128 = np.load("./winddata/step/24/s2smae.npy")/83000
rs2srmse128 = np.load("./winddata/step/24/s2srmse.npy")/83000
rvectormae128 = np.load("./winddata/step/24/vectormae.npy")/83000
rvectorrmse128 = np.load("./winddata/step/24/vectorrmse.npy")/83000
rpointmae128 = np.load("./winddata/step/24/singlemae.npy")/83000
rpointrmse128 = np.load("./winddata/step/24/singlermse.npy")/83000
ps2smae128 = np.load("./pricedata/step/24/s2smae.npy")/83000
ps2srmse128 = np.load("./pricedata/step/24/s2srmse.npy")/83000
pvectormae128 = np.load("./pricedata/step/24/vectormae.npy")/83000
pvectorrmse128 = np.load("./pricedata/step/24/vectorrmse.npy")/83000
ppointmae128 = np.load("./pricedata/step/24/singlemae.npy")/83000
ppointrmse128 = np.load("./pricedata/step/24/singlermse.npy")/83000
###############################################################################

axs[1,0].plot(x, rs2smae16, linewidth=2,marker = "*",markersize = 13, c = 'royalblue',label='# layer = 1')
axs[1,0].plot(x, rs2smae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# layer = 2')
axs[1,0].plot(x, rs2smae128, linewidth=2,marker = "^",markersize = 13, c = 'orange',label='# layer = 3')


axs[1,1].plot(x, rvectormae16, linewidth=2,marker = "*",markersize = 13, c = 'royalblue',label='# layer = 1')
axs[1,1].plot(x, rvectormae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# layer = 2')
axs[1,1].plot(x, rvectormae128, linewidth=2, marker = "^",markersize = 13,c = 'orange', label='# layer = 3')


axs[1,2].plot(x, rpointmae16, linewidth=2,marker = "*", markersize = 13,c = 'royalblue',label='# layer = 1')
axs[1,2].plot(x, rpointmae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# layer = 2')
axs[1,2].plot(x, rpointmae128, linewidth=2, marker = "^",markersize = 13,c = 'orange', label='# layer = 3')

###############################################################################
rs2smae16 = np.load("./winddata/layer/1/s2smae.npy")/83000
rs2srmse16 = np.load("./winddata/layer/1/s2srmse.npy")/83000
rvectormae16 = np.load("./winddata/layer/1/vectormae.npy")/83000
rvectorrmse16 = np.load("./winddata/layer/1/vectorrmse.npy")/83000
rpointmae16 = np.load("./winddata/layer/1/singlemae.npy")/83000
rpointrmse16 = np.load("./winddata/layer/1/singlermse.npy")/83000
ps2smae16 = np.load("./pricedata/layer/1/s2smae.npy")/83000
ps2srmse16 = np.load("./pricedata/layer/1/s2srmse.npy")/83000
pvectormae16 = np.load("./pricedata/layer/1/vectormae.npy")/83000
pvectorrmse16 = np.load("./pricedata/layer/1/vectorrmse.npy")/83000
ppointmae16 = np.load("./pricedata/layer/1/singlemae.npy")/83000
ppointrmse16 = np.load("./pricedata/layer/1/singlermse.npy")/83000
###############################################################################
rs2smae64 = np.load("./winddata/layer/2/s2smae.npy")/83000
rs2srmse64 = np.load("./winddata/layer/2/s2srmse.npy")/83000
rvectormae64 = np.load("./winddata/layer/2/vectormae.npy")/83000
rvectorrmse64 = np.load("./winddata/layer/2/vectorrmse.npy")/83000
rpointmae64 = np.load("./winddata/layer/2/singlemae.npy")/83000
rpointrmse64 = np.load("./winddata/layer/2/singlermse.npy")/83000
ps2smae64 = np.load("./pricedata/layer/2/s2smae.npy")/83000
ps2srmse64 = np.load("./pricedata/layer/2/s2srmse.npy")/83000
pvectormae64 = np.load("./pricedata/layer/2/vectormae.npy")/83000
pvectorrmse64 = np.load("./pricedata/layer/2/vectorrmse.npy")/83000
ppointmae64 = np.load("./pricedata/layer/2/singlemae.npy")/83000
ppointrmse64 = np.load("./pricedata/layer/2/singlermse.npy")/83000
###############################################################################
rs2smae128 = np.load("./winddata/layer/3/s2smae.npy")/83000
rs2srmse128 = np.load("./winddata/layer/3/s2srmse.npy")/83000
rvectormae128 = np.load("./winddata/layer/3/vectormae.npy")/83000
rvectorrmse128 = np.load("./winddata/layer/3/vectorrmse.npy")/83000
rpointmae128 = np.load("./winddata/layer/3/singlemae.npy")/83000
rpointrmse128 = np.load("./winddata/layer/3/singlermse.npy")/83000
ps2smae128 = np.load("./pricedata/layer/3/s2smae.npy")/83000
ps2srmse128 = np.load("./pricedata/layer/3/s2srmse.npy")/83000
pvectormae128 = np.load("./pricedata/layer/3/vectormae.npy")/83000
pvectorrmse128 = np.load("./pricedata/layer/3/vectorrmse.npy")/83000
ppointmae128 = np.load("./pricedata/layer/3/singlemae.npy")/83000
ppointrmse128 = np.load("./pricedata/layer/3/singlermse.npy")/83000
###############################################################################

axs[2,0].plot(x, rs2smae16, linewidth=2,marker = "*", markersize = 13,c = 'royalblue',label='# look-back steps = 6')
axs[2,0].plot(x, rs2smae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# look-back steps = 12')
axs[2,0].plot(x, rs2smae128, linewidth=2, marker = "^",markersize = 13,c = 'orange',label='# look-back steps = 24')


axs[2,1].plot(x, rvectormae16, linewidth=2,marker = "*",markersize = 13,c = 'royalblue', label='# look-back steps = 6')
axs[2,1].plot(x, rvectormae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# look-back steps = 12')
axs[2,1].plot(x, rvectormae128, linewidth=2,marker = "^",markersize = 13,c = 'orange', label='# look-back steps = 24')

axs[2,2].plot(x, rpointmae16, linewidth=2,marker = "*", markersize = 13,c = 'royalblue',label='# look-back steps = 6')
axs[2,2].plot(x, rpointmae64, linewidth=2,marker = "o",markersize = 13, c = 'gold',label='# look-back steps = 12')
axs[2,2].plot(x, rpointmae128, linewidth=2, marker = "^",markersize = 13, c = 'orange',label='# look-back steps = 24')


###############################################################################
axs[0,0].legend(fontsize=fs)
#axs[0,1].legend(fontsize=fs)
#axs[0,2].legend(fontsize=fs)

axs[1,0].legend(fontsize=fs)
#axs[1,1].legend(fontsize=fs)
#axs[1,2].legend(fontsize=fs)

axs[2,0].legend(fontsize=fs)
#axs[2,1].legend(fontsize=fs)
#axs[2,2].legend(fontsize=fs)

axs[0,0].tick_params(labelsize=fs2)
axs[0,1].tick_params(labelsize=fs2) 
axs[0,2].tick_params(labelsize=fs2) 

axs[1,0].tick_params(labelsize=fs2) 
axs[1,1].tick_params(labelsize=fs2) 
axs[1,2].tick_params(labelsize=fs2) 

axs[2,0].tick_params(labelsize=fs2)
axs[2,1].tick_params(labelsize=fs2)
axs[2,2].tick_params(labelsize=fs2)

axs[0,2].set_ylim(0, 0.1)
axs[1,2].set_ylim(0, 0.1)
axs[2,2].set_ylim(0, 0.1)

axs[0,0].set_title('# Hidden state, SPTA', fontsize=fs1)
axs[0,1].set_title('# Hidden state, VPTA', fontsize=fs1)
axs[0,2].set_title('# Hidden state, PPTA', fontsize=fs1)
axs[1,0].set_title('# Layers, SPTA', fontsize=fs1)
axs[1,1].set_title('# Layers, VPTA', fontsize=fs1)
axs[1,2].set_title('# Layers, PPTA', fontsize=fs1)
axs[2,0].set_title('# Look-back steps, SPTA', fontsize=fs1)
axs[2,1].set_title('# Look-back steps, VPTA', fontsize=fs1)
axs[2,2].set_title('# Look-back steps, PPTA', fontsize=fs1)
   

plt.savefig("./figure/winderror.pdf",dpi=1600)
###############################################################################



###############################################################################
rs2smae16 = np.load("./winddata/state/16/s2smae.npy")/210
rs2srmse16 = np.load("./winddata/state/16/s2srmse.npy")/210
rvectormae16 = np.load("./winddata/state/16/vectormae.npy")/210
rvectorrmse16 = np.load("./winddata/state/16/vectorrmse.npy")/210
rpointmae16 = np.load("./winddata/state/16/singlemae.npy")/210
rpointrmse16 = np.load("./winddata/state/16/singlermse.npy")/210
ps2smae16 = np.load("./pricedata/state/16/s2smae.npy")/210
ps2srmse16 = np.load("./pricedata/state/16/s2srmse.npy")/210
pvectormae16 = np.load("./pricedata/state/16/vectormae.npy")/210
pvectorrmse16 = np.load("./pricedata/state/16/vectorrmse.npy")/210
ppointmae16 = np.load("./pricedata/state/16/singlemae.npy")/210
ppointrmse16 = np.load("./pricedata/state/16/singlermse.npy")/210
###############################################################################
rs2smae64 = np.load("./winddata/state/64/s2smae.npy")/210
rs2srmse64 = np.load("./winddata/state/64/s2srmse.npy")/210
rvectormae64 = np.load("./winddata/state/64/vectormae.npy")/210
rvectorrmse64 = np.load("./winddata/state/64/vectorrmse.npy")/210
rpointmae64 = np.load("./winddata/state/64/singlemae.npy")/210
rpointrmse64 = np.load("./winddata/state/64/singlermse.npy")/210
ps2smae64 = np.load("./pricedata/state/64/s2smae.npy")/210
ps2srmse64 = np.load("./pricedata/state/64/s2srmse.npy")/210
pvectormae64 = np.load("./pricedata/state/64/vectormae.npy")/210
pvectorrmse64 = np.load("./pricedata/state/64/vectorrmse.npy")/210
ppointmae64 = np.load("./pricedata/state/64/singlemae.npy")/210
ppointrmse64 = np.load("./pricedata/state/64/singlermse.npy")/210
###############################################################################
rs2smae128 = np.load("./winddata/state/128/s2smae.npy")/210
rs2srmse128 = np.load("./winddata/state/128/s2srmse.npy")/210
rvectormae128 = np.load("./winddata/state/128/vectormae.npy")/210
rvectorrmse128 = np.load("./winddata/state/128/vectorrmse.npy")/210
rpointmae128 = np.load("./winddata/state/128/singlemae.npy")/210
rpointrmse128 = np.load("./winddata/state/128/singlermse.npy")/210
ps2smae128 = np.load("./pricedata/state/128/s2smae.npy")/210
ps2srmse128 = np.load("./pricedata/state/128/s2srmse.npy")/210
pvectormae128 = np.load("./pricedata/state/128/vectormae.npy")/210
pvectorrmse128 = np.load("./pricedata/state/128/vectorrmse.npy")/210
ppointmae128 = np.load("./pricedata/state/128/singlemae.npy")/210
ppointrmse128 = np.load("./pricedata/state/128/singlermse.npy")/210
###############################################################################

x = [1,10,20,30,40]
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12,9), sharey=True, constrained_layout=True)
axs[0,0].plot(x, ps2smae16, linewidth=2,marker = "*", markersize = 13,c = 'royalblue', label='# hidden state = 16')
axs[0,0].plot(x, ps2smae64, linewidth=2,marker = "o", markersize = 13,c = 'gold',label='# hidden state = 64')
axs[0,0].plot(x, ps2smae128, linewidth=2, marker = "^",markersize = 13, c='orange', label='# hidden state = 128')


axs[0,1].plot(x, pvectormae16, linewidth=2,marker = "*",markersize = 13,c = 'royalblue', label='# hidden state = 16')
axs[0,1].plot(x, pvectormae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# hidden state = 64')
axs[0,1].plot(x, pvectormae128, linewidth=2, marker = "^",markersize = 13, c='orange', label='# hidden state = 128')


axs[0,2].plot(x, ppointmae16, linewidth=2,marker = "*",markersize = 13, c = 'royalblue', label='# hidden state = 16')
axs[0,2].plot(x, ppointmae64, linewidth=2,marker = "o", markersize = 13,c = 'gold',label='# hidden state = 64')
axs[0,2].plot(x, ppointmae128, linewidth=2, marker = "^",markersize = 13,  c='orange', label='# hidden state = 128')


###############################################################################
rs2smae16 = np.load("./winddata/layer/1/s2smae.npy")/210
rs2srmse16 = np.load("./winddata/layer/1/s2srmse.npy")/210
rvectormae16 = np.load("./winddata/layer/1/vectormae.npy")/210
rvectorrmse16 = np.load("./winddata/layer/1/vectorrmse.npy")/210
rpointmae16 = np.load("./winddata/layer/1/singlemae.npy")/210
rpointrmse16 = np.load("./winddata/layer/1/singlermse.npy")/210
ps2smae16 = np.load("./pricedata/layer/1/s2smae.npy")/210
ps2srmse16 = np.load("./pricedata/layer/1/s2srmse.npy")/210
pvectormae16 = np.load("./pricedata/layer/1/vectormae.npy")/210
pvectorrmse16 = np.load("./pricedata/layer/1/vectorrmse.npy")/210
ppointmae16 = np.load("./pricedata/layer/1/singlemae.npy")/210
ppointrmse16 = np.load("./pricedata/layer/1/singlermse.npy")/210
###############################################################################
rs2smae64 = np.load("./winddata/layer/2/s2smae.npy")/210
rs2srmse64 = np.load("./winddata/layer/2/s2srmse.npy")/210
rvectormae64 = np.load("./winddata/layer/2/vectormae.npy")/210
rvectorrmse64 = np.load("./winddata/layer/2/vectorrmse.npy")/210
rpointmae64 = np.load("./winddata/layer/2/singlemae.npy")/210
rpointrmse64 = np.load("./winddata/layer/2/singlermse.npy")/210
ps2smae64 = np.load("./pricedata/layer/2/s2smae.npy")/210
ps2srmse64 = np.load("./pricedata/layer/2/s2srmse.npy")/210
pvectormae64 = np.load("./pricedata/layer/2/vectormae.npy")/210
pvectorrmse64 = np.load("./pricedata/layer/2/vectorrmse.npy")/210
ppointmae64 = np.load("./pricedata/layer/2/singlemae.npy")/210
ppointrmse64 = np.load("./pricedata/layer/2/singlermse.npy")/210
###############################################################################
rs2smae128 = np.load("./winddata/layer/3/s2smae.npy")/210
rs2srmse128 = np.load("./winddata/layer/3/s2srmse.npy")/210
rvectormae128 = np.load("./winddata/layer/3/vectormae.npy")/210
rvectorrmse128 = np.load("./winddata/layer/3/vectorrmse.npy")/210
rpointmae128 = np.load("./winddata/layer/3/singlemae.npy")/210
rpointrmse128 = np.load("./winddata/layer/3/singlermse.npy")/210
ps2smae128 = np.load("./pricedata/layer/3/s2smae.npy")/210
ps2srmse128 = np.load("./pricedata/layer/3/s2srmse.npy")/210
pvectormae128 = np.load("./pricedata/layer/3/vectormae.npy")/210
pvectorrmse128 = np.load("./pricedata/layer/3/vectorrmse.npy")/210
ppointmae128 = np.load("./pricedata/layer/3/singlemae.npy")/210
ppointrmse128 = np.load("./pricedata/layer/3/singlermse.npy")/210
###############################################################################

axs[1,0].plot(x, ps2smae16, linewidth=2,marker = "*", markersize = 13,c = 'royalblue',label='# layer = 1')
axs[1,0].plot(x, ps2smae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# layer = 2')
axs[1,0].plot(x, ps2smae128, linewidth=2,  marker = "^",markersize = 13,c = 'orange', label='# layer = 3')


axs[1,1].plot(x, pvectormae16, linewidth=2,marker = "*",markersize = 13, c = 'royalblue',label='# layer = 1')
axs[1,1].plot(x, pvectormae64, linewidth=2,marker = "o",markersize = 13,c = 'gold', label='# layer = 2')
axs[1,1].plot(x, pvectormae128, linewidth=2,  marker = "^",markersize = 13,c = 'orange', label='# layer = 3')


axs[1,2].plot(x, ppointmae16, linewidth=2,marker = "*",markersize = 13,c = 'royalblue', label='# layer = 1')
axs[1,2].plot(x, ppointmae64, linewidth=2, marker = "o",markersize = 13,c = 'gold',label='# layer = 2')
axs[1,2].plot(x, ppointmae128, linewidth=2, marker = "^",markersize = 13, c = 'orange', label='# layer = 3')


###############################################################################
rs2smae16 = np.load("./winddata/step/6/s2smae.npy")/210
rs2srmse16 = np.load("./winddata/step/6/s2srmse.npy")/210
rvectormae16 = np.load("./winddata/step/6/vectormae.npy")/210
rvectorrmse16 = np.load("./winddata/step/6/vectorrmse.npy")/210
rpointmae16 = np.load("./winddata/step/6/singlemae.npy")/210
rpointrmse16 = np.load("./winddata/step/6/singlermse.npy")/210
ps2smae16 = np.load("./pricedata/step/6/s2smae.npy")/210
ps2srmse16 = np.load("./pricedata/step/6/s2srmse.npy")/210
pvectormae16 = np.load("./pricedata/step/6/vectormae.npy")/210
pvectorrmse16 = np.load("./pricedata/step/6/vectorrmse.npy")/210
ppointmae16 = np.load("./pricedata/step/6/singlemae.npy")/210
ppointrmse16 = np.load("./pricedata/step/6/singlermse.npy")/210
###############################################################################
rs2smae64 = np.load("./winddata/step/12/s2smae.npy")/210
rs2srmse64 = np.load("./winddata/step/12/s2srmse.npy")/210
rvectormae64 = np.load("./winddata/step/12/vectormae.npy")/210
rvectorrmse64 = np.load("./winddata/step/12/vectorrmse.npy")/210
rpointmae64 = np.load("./winddata/step/12/singlemae.npy")/210
rpointrmse64 = np.load("./winddata/step/12/singlermse.npy")/210
ps2smae64 = np.load("./pricedata/step/12/s2smae.npy")/210
ps2srmse64 = np.load("./pricedata/step/12/s2srmse.npy")/210
pvectormae64 = np.load("./pricedata/step/12/vectormae.npy")/210
pvectorrmse64 = np.load("./pricedata/step/12/vectorrmse.npy")/210
ppointmae64 = np.load("./pricedata/step/12/singlemae.npy")/210
ppointrmse64 = np.load("./pricedata/step/12/singlermse.npy")/210
###############################################################################
rs2smae128 = np.load("./winddata/step/24/s2smae.npy")/210
rs2srmse128 = np.load("./winddata/step/24/s2srmse.npy")/210
rvectormae128 = np.load("./winddata/step/24/vectormae.npy")/210
rvectorrmse128 = np.load("./winddata/step/24/vectorrmse.npy")/210
rpointmae128 = np.load("./winddata/step/24/singlemae.npy")/210
rpointrmse128 = np.load("./winddata/step/24/singlermse.npy")/210
ps2smae128 = np.load("./pricedata/step/24/s2smae.npy")/210
ps2srmse128 = np.load("./pricedata/step/24/s2srmse.npy")/210
pvectormae128 = np.load("./pricedata/step/24/vectormae.npy")/210
pvectorrmse128 = np.load("./pricedata/step/24/vectorrmse.npy")/210
ppointmae128 = np.load("./pricedata/step/24/singlemae.npy")/210
ppointrmse128 = np.load("./pricedata/step/24/singlermse.npy")/210
###############################################################################

axs[2,0].plot(x, ps2smae16, linewidth=2,marker = "*", markersize = 13,c= 'royalblue',label='# look-back steps = 6')
axs[2,0].plot(x, ps2smae64, linewidth=2, marker = "o",markersize = 13, c='gold',label='# look-back steps = 12')
axs[2,0].plot(x, ps2smae128, linewidth=2,marker = "^", markersize = 13,c='orange',label='# look-back steps = 24')

axs[2,1].plot(x, pvectormae16, linewidth=2,marker = "*",markersize = 13, c= 'royalblue',label='# look-back steps = 6')
axs[2,1].plot(x, pvectormae64, linewidth=2, marker = "o",markersize = 13, c='gold', label='# look-back steps = 12')
axs[2,1].plot(x, pvectormae128, linewidth=2, marker = "^",markersize = 13, c='orange',label='# look-back steps = 24')

axs[2,2].plot(x, ppointmae16, linewidth=2,marker = "*",markersize = 13,c= 'royalblue', label='# look-back steps = 6')
axs[2,2].plot(x, ppointmae64, linewidth=2, marker = "o",markersize = 13, c='gold',label='# look-back steps = 12')
axs[2,2].plot(x, ppointmae128, linewidth=2, marker = "^",markersize = 13, c='orange',label='# look-back steps = 24')


###############################################################################
axs[0,0].legend(fontsize=fs)
#axs[0,1].legend(fontsize=fs)
#axs[0,2].legend(fontsize=fs)

axs[1,0].legend(fontsize=fs)
#axs[1,1].legend(fontsize=fs)
#axs[1,2].legend(fontsize=fs)

axs[2,0].legend(fontsize=fs)
#axs[2,1].legend(fontsize=fs)
#axs[2,2].legend(fontsize=fs)

axs[0,0].tick_params(labelsize=fs2)
axs[0,1].tick_params(labelsize=fs2) 
axs[0,2].tick_params(labelsize=fs2) 

axs[1,0].tick_params(labelsize=fs2) 
axs[1,1].tick_params(labelsize=fs2) 
axs[1,2].tick_params(labelsize=fs2) 

axs[2,0].tick_params(labelsize=fs2)
axs[2,1].tick_params(labelsize=fs2)
axs[2,2].tick_params(labelsize=fs2)

axs[0,0].set_title('# Hidden state, SPTA', fontsize=fs1)
axs[0,1].set_title('# Hidden state, VPTA', fontsize=fs1)
axs[0,2].set_title('# Hidden state, PPTA', fontsize=fs1)
axs[1,0].set_title('# Layers, SPTA', fontsize=fs1)
axs[1,1].set_title('# Layers, VPTA', fontsize=fs1)
axs[1,2].set_title('# Layers, PPTA', fontsize=fs1)
axs[2,0].set_title('# Look-back steps, SPTA', fontsize=fs1)
axs[2,1].set_title('# Look-back steps, VPTA', fontsize=fs1)
axs[2,2].set_title('# Look-back steps, PPTA', fontsize=fs1)

plt.savefig("./figure/priceerror.pdf",dpi=1600)
