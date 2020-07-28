# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 00:45:54 2020

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

"""
###############################################################################################
k=34; fs=17
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharey=True)
sns.kdeplot(s2sonl[0,:k]/s2sofl[0,:k], label='DETA', ax = axs[0,0])
sns.kdeplot(s2slstm[0,:k]/s2sofl[0,:k],  label='SPTA', ax = axs[0,0])
sns.kdeplot(vectorlstm[0,:k]/s2sofl[0,:k],  label='VPTA',ax = axs[0,0])
sns.kdeplot(pointlstm[0,:k]/s2sofl[0,:k],  label='PPTA',ax = axs[0,0])

#axs.grid(True, linestyle='-.')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[0,0].legend()
axs[0,0].set_title('# month = 1, training', fontsize=fs)


sns.kdeplot(s2sonl1[0,:k]/s2sofl1[0,:k], label='DETA', ax = axs[0,1])
sns.kdeplot(s2slstm1[0,:k]/s2sofl1[0,:k],  label='SPTA', ax = axs[0,1])
sns.kdeplot(vectorlstm1[0,:k]/s2sofl1[0,:k], label='VPTA',ax = axs[0,1])
sns.kdeplot(pointlstm1[0,:k]/s2sofl1[0,:k], label='PPTA',ax = axs[0,1])

#axs.grid(True, linestyle='-.')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[0,1].legend()
axs[0,1].set_title('# month = 1, testing', fontsize=fs)


robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

sns.kdeplot(robdeta, label='DETA', ax = axs[0,2])
sns.kdeplot(robs2s,  label='SPTA', ax = axs[0,2])
sns.kdeplot(robvector, label='VPTA',ax = axs[0,2])
sns.kdeplot(robpoint, label='PPTA',ax = axs[0,2])

#axs.grid(True, linestyle='-.')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[0,2].legend()
axs[0,2].set_title('# month = 1, robustness', fontsize=fs)


#####################################################################################################
k=10
sns.kdeplot(s2sonl[0,:k]/s2sofl[0,:k], label='DETA', ax = axs[1,0])
sns.kdeplot(s2slstm[0,:k]/s2sofl[0,:k],  label='SPTA', ax = axs[1,0])
sns.kdeplot(vectorlstm[0,:k]/s2sofl[0,:k],  label='VPTA',ax = axs[1,0])
sns.kdeplot(pointlstm[0,:k]/s2sofl[0,:k],  label='PPTA',ax = axs[1,0])

#axs.grid(True, linestyle='-.')
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[1,0].legend()
axs[1,0].set_title('# month = 3, training', fontsize=fs)

sns.kdeplot(s2sonl1[0,:k]/s2sofl1[0,:k], label='DETA', ax = axs[1,1])
sns.kdeplot(s2slstm1[0,:k]/s2sofl1[0,:k],  label='SPTA', ax = axs[1,1])
sns.kdeplot(vectorlstm1[0,:k]/s2sofl1[0,:k], label='VPTA',ax = axs[1,1])
sns.kdeplot(pointlstm1[0,:k]/s2sofl1[0,:k], label='PPTA',ax = axs[1,1])

#axs.grid(True, linestyle='-.')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[1,1].legend()
axs[1,1].set_title('# month = 3, testing', fontsize=fs)

robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

sns.kdeplot(robdeta, label='DETA', ax = axs[1,2])
sns.kdeplot(robs2s,  label='SPTA', ax = axs[1,2])
sns.kdeplot(robvector, label='VPTA',ax = axs[1,2])
sns.kdeplot(robpoint, label='PPTA',ax = axs[1,2])

#axs.grid(True, linestyle='-.')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[1,2].legend()
axs[1,2].set_title('# month = 3, robustness', fontsize=fs)

#############################################################################################

k=4

sns.kdeplot(s2sonl[0,:k]/s2sofl[0,:k], label='DETA', ax = axs[2,0])
sns.kdeplot(s2slstm[0,:k]/s2sofl[0,:k],  label='SPTA', ax = axs[2,0])
sns.kdeplot(vectorlstm[0,:k]/s2sofl[0,:k],  label='VPTA',ax = axs[2,0])
sns.kdeplot(pointlstm[0,:k]/s2sofl[0,:k],  label='PPTA',ax = axs[2,0])

#axs.grid(True, linestyle='-.')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[2,0].legend()
axs[2,0].set_title('# month = 6, training', fontsize=fs)


sns.kdeplot(s2sonl1[0,:k]/s2sofl1[0,:k], label='DETA', ax = axs[2,1])
sns.kdeplot(s2slstm1[0,:k]/s2sofl1[0,:k],  label='SPTA', ax = axs[2,1])
sns.kdeplot(vectorlstm1[0,:k]/s2sofl1[0,:k], label='VPTA',ax = axs[2,1])
sns.kdeplot(pointlstm1[0,:k]/s2sofl1[0,:k], label='PPTA',ax = axs[2,1])

#axs.grid(True, linestyle='-.')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[2,1].legend()
axs[2,1].set_title('# month = 6, testing', fontsize=fs)

robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]


sns.kdeplot(robdeta, label='DETA', ax = axs[2,2])
sns.kdeplot(robs2s,  label='SPTA', ax = axs[2,2])
sns.kdeplot(robvector, label='VPTA',ax = axs[2,2])
sns.kdeplot(robpoint, label='PPTA',ax = axs[2,2])

#axs.grid(True, linestyle='-.')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
axs[2,2].legend()
axs[2,2].set_title('# month = 6, robustness', fontsize=fs)
plt.savefig("./figure/rob1.pdf")
##################################################################################
"""

fs=18; line = 1.5
###############################################################################################
k=34
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15, 12), sharey=True)
sns.kdeplot(s2sonl[0,:k]/s2sofl[0,:k], c = "royalblue",linewidth = line, label='# month = 1', ax = axs[0,0])
sns.kdeplot(s2slstm[0,:k]/s2sofl[0,:k], c = "royalblue", linewidth = line,label='# month = 1', ax = axs[0,1])
sns.kdeplot(vectorlstm[0,:k]/s2sofl[0,:k], c = "royalblue",linewidth = line,  label='# month = 1',ax = axs[0,2])
sns.kdeplot(pointlstm[0,:k]/s2sofl[0,:k], c = "royalblue",linewidth = line, label='# month = 1',ax = axs[0,3])

sns.kdeplot(s2sonl1[0,:k]/s2sofl1[0,:k], c = "royalblue",linewidth = line,label='# month = 1', ax = axs[1,0])
sns.kdeplot(s2slstm1[0,:k]/s2sofl1[0,:k],  c = "royalblue",linewidth = line,label='# month = 1', ax = axs[1,1])
sns.kdeplot(vectorlstm1[0,:k]/s2sofl1[0,:k],c = "royalblue",linewidth = line, label='# month = 1',ax = axs[1,2])
sns.kdeplot(pointlstm1[0,:k]/s2sofl1[0,:k], c = "royalblue",linewidth = line,label='# month = 1',ax = axs[1,3])

robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

sns.kdeplot(robdeta, label='# month = 1',c = "royalblue",linewidth = line, ax = axs[2,0])
sns.kdeplot(robs2s,  label='# month = 1', c = "royalblue",linewidth = line,ax = axs[2,1])
sns.kdeplot(robvector, label='# month = 1',c = "royalblue",linewidth = line,ax = axs[2,2])
sns.kdeplot(robpoint, label='# month = 1',c = "royalblue",linewidth = line,ax = axs[2,3])

#####################################################################################################
k=10
sns.kdeplot(s2sonl[0,:k]/s2sofl[0,:k], c = "gold", linewidth = line, label='# month = 3', ax = axs[0,0])
sns.kdeplot(s2slstm[0,:k]/s2sofl[0,:k], c = "gold", linewidth = line,label='# month = 3', ax = axs[0,1])
sns.kdeplot(vectorlstm[0,:k]/s2sofl[0,:k], c = "gold", linewidth = line,label='# month = 3',ax = axs[0,2])
sns.kdeplot(pointlstm[0,:k]/s2sofl[0,:k], c = "gold",linewidth = line, label='# month = 3',ax = axs[0,3])

sns.kdeplot(s2sonl1[0,:k]/s2sofl1[0,:k], c = "gold",linewidth = line,label='# month = 3', ax = axs[1,0])
sns.kdeplot(s2slstm1[0,:k]/s2sofl1[0,:k], c = "gold", linewidth = line,label='# month = 3', ax = axs[1,1])
sns.kdeplot(vectorlstm1[0,:k]/s2sofl1[0,:k], c = "gold",linewidth = line,label='# month = 3',ax = axs[1,2])
sns.kdeplot(pointlstm1[0,:k]/s2sofl1[0,:k],c = "gold",linewidth = line, label='# month = 3',ax = axs[1,3])

robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

sns.kdeplot(robdeta, label='# month = 3',c = "gold",linewidth = line, ax = axs[2,0])
sns.kdeplot(robs2s,  label='# month = 3', c = "gold",linewidth = line,ax = axs[2,1])
sns.kdeplot(robvector, label='# month = 3',c = "gold",linewidth = line,ax = axs[2,2])
sns.kdeplot(robpoint, label='# month = 3',c = "gold",linewidth = line,ax = axs[2,3])

#############################################################################################
k=4
sns.kdeplot(s2sonl[0,:k]/s2sofl[0,:k], c = "orange", linewidth = line, label='# month = 6', ax = axs[0,0])
sns.kdeplot(s2slstm[0,:k]/s2sofl[0,:k], c = "orange", linewidth = line, label='# month = 6', ax = axs[0,1])
sns.kdeplot(vectorlstm[0,:k]/s2sofl[0,:k], c = "orange", linewidth = line, label='# month = 6',ax = axs[0,2])
sns.kdeplot(pointlstm[0,:k]/s2sofl[0,:k], c = "orange", linewidth = line, label='# month = 6',ax = axs[0,3])

sns.kdeplot(s2sonl1[0,:k]/s2sofl1[0,:k],c = "orange", linewidth = line, label='# month = 6', ax = axs[1,0])
sns.kdeplot(s2slstm1[0,:k]/s2sofl1[0,:k], c = "orange", linewidth = line, label='# month = 6', ax = axs[1,1])
sns.kdeplot(vectorlstm1[0,:k]/s2sofl1[0,:k], c = "orange", linewidth = line,label='# month = 6',ax = axs[1,2])
sns.kdeplot(pointlstm1[0,:k]/s2sofl1[0,:k],c = "orange", linewidth = line, label='# month = 6',ax = axs[1,3])

robdeta = s2sonl1[0,:k]/s2sofl1[0,:k]-s2sonl[0,:k]/s2sofl[0,:k]
robs2s = s2slstm1[0,:k]/s2sofl1[0,:k]-s2slstm[0,:k]/s2sofl[0,:k]
robvector =vectorlstm1[0,:k]/s2sofl1[0,:k]-vectorlstm[0,:k]/s2sofl[0,:k]
robpoint = pointlstm1[0,:k]/pointofl1[0,:k]-pointlstm[0,:k]/s2sofl[0,:k]

sns.kdeplot(robdeta,c = "orange", linewidth = line, label='# month = 6', ax = axs[2,0])
sns.kdeplot(robs2s, c = "orange", linewidth = line, label='# month = 6', ax = axs[2,1])
sns.kdeplot(robvector,c = "orange", linewidth = line, label='# month = 6',ax = axs[2,2])
sns.kdeplot(robpoint,c = "orange", linewidth = line, label='# month = 6',ax = axs[2,3])

fs = 18
axs[0,0].legend(fontsize=fs)
axs[0,1].legend(fontsize=fs)
axs[0,2].legend(fontsize=fs)
axs[0,3].legend(fontsize=fs)
axs[1,0].legend(fontsize=fs)
axs[1,1].legend(fontsize=fs)
axs[1,2].legend(fontsize=fs)
axs[1,3].legend(fontsize=fs)
axs[2,0].legend(fontsize=fs)
axs[2,1].legend(fontsize=fs)
axs[2,2].legend(fontsize=fs)
axs[2,3].legend(fontsize=fs)

fs2 = 17
axs[0,0].tick_params(labelsize=fs2)
axs[0,1].tick_params(labelsize=fs2) 
axs[0,2].tick_params(labelsize=fs2)
axs[0,3].tick_params(labelsize=fs2) 

axs[1,0].tick_params(labelsize=fs2) 
axs[1,1].tick_params(labelsize=fs2) 
axs[1,2].tick_params(labelsize=fs2)
axs[1,3].tick_params(labelsize=fs2)  

axs[2,0].tick_params(labelsize=fs2)
axs[2,1].tick_params(labelsize=fs2)
axs[2,2].tick_params(labelsize=fs2)
axs[2,3].tick_params(labelsize=fs2)  

axs[0,0].set_xlim([1.0,1.1])
axs[0,1].set_xlim([1.0,1.1])
axs[0,2].set_xlim([1.0,1.1])
axs[0,3].set_xlim([1.0,1.1])
axs[1,0].set_xlim([1.0,1.1])
axs[1,1].set_xlim([1.0,1.1])
axs[1,2].set_xlim([1.0,1.1])
axs[1,3].set_xlim([1.0,1.1])
axs[2,0].set_xlim([-0.1,0.1])
axs[2,1].set_xlim([-0.1,0.1])
axs[2,2].set_xlim([-0.1,0.1])
axs[2,3].set_xlim([-0.1,0.1])

fs = 20
axs[0,0].set_title('Traning, DETA', fontsize=fs)
axs[0,1].set_title('Traning, SETA', fontsize=fs)
axs[0,2].set_title('Traning, VETA', fontsize=fs)
axs[0,3].set_title('Traning, PETA', fontsize=fs)
axs[1,0].set_title('Testing, DETA', fontsize=fs)
axs[1,1].set_title('Testing, SETA', fontsize=fs)
axs[1,2].set_title('Testing, VETA', fontsize=fs)
axs[1,3].set_title('Testing, PETA', fontsize=fs)
axs[2,0].set_title('Robustness, DETA', fontsize=fs)
axs[2,1].set_title('Robustness, SETA', fontsize=fs)
axs[2,2].set_title('Robustness, VETA', fontsize=fs)
axs[2,3].set_title('Robustness, PETA', fontsize=fs)

fig.tight_layout()
plt.savefig("./figure/rob2.pdf")
##################################################################################