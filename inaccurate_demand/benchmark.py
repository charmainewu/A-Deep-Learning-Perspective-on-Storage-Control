# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
from sklearn import mixture
import math
from RL_brain import QLearningTable
import warnings
warnings.filterwarnings("ignore")

class BenchmarkInAccurateDemand:

    def __init__(self, Capacity, Window, Ntrain,Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM):
      self.B = Capacity
      self.W = Window
      self.P = Price
      self.D = Demand
      self.R = Renewables
      self.N = Ntrain
      self.T = Ntest
      self.V = Nvalid
      self.I = Interval
      self.BD = Backday
      self.PM = PriceModel
      self.WM = WindModel
      self.PS = PriceScalar
      self.WS = WindScalar
      self.WSD = WindStackData
      self.PSD = PriceStackData
      self.LSTM = LSTM

    def omg(self):
        LAMDA = 1;
        P_MIN = min(self.P); P_MAX = max(self.P);
        S_MAX = self.B; S_MIN = 0
        U_MAX= self.B; U_MIN = -self.B
        DG_UP = P_MAX; DG_DOWN = P_MIN
        
        W =( (S_MAX-S_MIN)-(U_MAX-U_MIN) )/(DG_UP-DG_DOWN)
        TAU = (DG_DOWN*(S_MIN-U_MIN)-DG_UP*(S_MAX-U_MAX))/(LAMDA*(DG_UP-DG_DOWN))
        
        u = np.zeros(len(self.P))
        s = np.zeros(len(self.P))
        
        for t in range(len(self.P)):
            c = np.r_[s[t]+TAU-W*self.P[t]]
            res = linprog(c,bounds=(-self.B,self.B))
            u[t] = res.x
            try:
                s[t+1] = s[t]+u[t]
            except:
                continue
        return u
      
    def stepto(self, action,observation,step,p,pbar,d):
        length = len(p)
        pbar = (1-0.7)*pbar+0.7*observation[0]
        if action == 0:
            reward = 0
            sl = observation[2]
            ch = 0
            dh = 0
            gh = observation[1]
            stepcost = observation[0]*observation[1]
        elif action == 1:
            reward = (observation[0]-pbar)*min(observation[2],observation[1])
            sl = observation[2]-min(observation[2],observation[1])
            ch = 0
            dh = min(observation[2],observation[1])
            gh = observation[1]-min(observation[2],observation[1])
            stepcost = observation[0]*(observation[1]-min(observation[2],observation[1]))
        elif action == 2:
            reward = (pbar-observation[0])*int((self.B-observation[2]))
            sl = observation[2]+int((self.B-observation[2]))
            ch = int((self.B-observation[2]))
            dh = 0
            gh = observation[1]
            stepcost = observation[0]*(observation[1]+int((self.B-observation[2])))
        
        if step == length-1:
            done = True
            observation_ =  'terminal'
            return observation_, reward, done, pbar, stepcost, sl, ch, dh, gh
        else:
            done = False
            observation_ =  np.array([p[step+1],d[step+1],sl])
            return observation_, reward, done, pbar, stepcost, sl, ch, dh, gh

    def train(self, d,p,RL):
        d = np.array(d)
        d = d.reshape(self.N)
        p = np.array(p)
        p = p.reshape(self.N)
        level = np.array([10,10000,10000])
        for episode in range(20):
            s = 0; step = 0; pbar = p[0]
            observation = np.array([p[0],d[0],s])
            while True:
                temp_ob = observation.copy()/level; temp_ob = temp_ob.astype(int);
                action = RL.choose_action(str(temp_ob))
                observation_, reward, done, pbar, stepcost,sl, ch, dh, gh = self.stepto(action,observation,step,p,pbar,d)
                if observation_== 'terminal':
                    RL.learn(str(temp_ob), action, reward, observation_)
                    print('terminal episode '+str(episode))
                    break
                else:
                    temp_ob_ = observation_.copy()/level; temp_ob_ = temp_ob_.astype(int);
                    RL.learn(str(temp_ob), action, reward, str(temp_ob_))
                observation = observation_
                step = step + 1
                if step>=(len(d)):
                    break
        return RL
    
    def rl(self):
        RL = QLearningTable(actions=list(range(3)))
        RL = self.train(self.D[:self.N],self.P[:self.N],RL)
        level = np.array([10,10000,10000])
        n_interval = int(self.T/self.I); 
        cost_rl = np.zeros(n_interval)
        for n in range(1, n_interval+1):
            a_real = self.D[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            d_real = (a_real - r_real)
            
            d_real = d_real.astype(int)
            
            d_real = d_real.reshape(len(d_real))
            p_real = p_real.reshape(len(p_real))
            
            s = 0; step = 0;pbar = p_real[0]
            observation = np.array([p_real[0],d_real[0],s])
            while True:
                temp_ob = observation.copy()/level; temp_ob = temp_ob.astype(int);
                action = RL.choose_action(str(temp_ob))
                observation_, reward, done, pbar, stepcost,sl, cd, dd, gd = self.stepto(action,observation,step,p_real,pbar,d_real)
                if observation_== 'terminal':
                    cost_rl[n-1] = cost_rl[n-1] + stepcost
                    break
                else:
                    cost_rl[n-1] = cost_rl[n-1] + stepcost
                observation = observation_
                step = step + 1
                if step>=self.I:
                    break
        cost_rl_copy = cost_rl.copy()
        for i in range(len(cost_rl_copy)):
          cost_rl[i] = sum(cost_rl_copy[:i+1])
        return cost_rl
      
    def solvelp(self, pw,dw,x0):
        #####objective function########
        c = np.r_[-pw,pw,np.zeros(len(pw))]
        #####constraints for capacity##
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a1[i] = -1
            a2[i] = 1
            if i == 0:
                a3[i] = -1
            else:
                a3[i] = -1
                a3[i-1] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_e =  np.array([a])
            else:
                A_e =  np.r_[A_e,[a]]
        b_e = np.zeros(len(pw)); b_e[0] = -x0
        
        #####constraints for upper#####
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a3[i] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_u1 =  np.array([a])
            else:
                A_u1 =  np.r_[A_u1,[a]]
        b_u1 = np.zeros(len(pw))+self.B
        
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a1[i] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_u2 =  np.array([a])
            else:
                A_u2 =  np.r_[A_u2,[a]]
        b_u2 = np.zeros(len(pw))+self.B
        
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a2[i] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_u3 =  np.array([a])
            else:
                A_u3 =  np.r_[A_u3,[a]]
        b_u3 = np.zeros(len(pw))+self.B
        
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a1[i] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_u4 =  np.array([a])
            else:
                A_u4 =  np.r_[A_u4,[a]]
        b_u4 = np.zeros(len(pw))+dw
        
        A_u = np.r_[A_u1,A_u2]
        A_u = np.r_[A_u,A_u3]
        A_u = np.r_[A_u,A_u4]
        
        b_u = np.r_[b_u1,b_u2]
        b_u = np.r_[b_u,b_u3]
        b_u = np.r_[b_u,b_u4]
        ################################
        res = linprog(c, A_ub=A_u, b_ub=b_u, A_eq=A_e, b_eq = b_e)
        return res.x
    
    def Ampc_hat(self, d,p,TS,TD,x0):
        T = len(p); W=3
        cost = np.zeros(T)
        dc = np.zeros(T)
        cc = np.zeros(T)
        xc = np.zeros(T)
        gc = np.zeros(T)
        xcc = x0
        for t in range(T):
            try:
              pred_price = self.pred_step(t,TS,TD,"Price")
              pred_price = pred_price.reshape(len(pred_price))
            except:
              pred_price = []
            if t+W<=T:
                pw = np.r_[np.array(p[t]),pred_price[:W-1]]
                dw = d[t:t+W]
                dw = dw.reshape(W)
                x = self.solvelp(pw,dw,xcc)
                dc[t] = x[0]
                cc[t] = x[W]
                xc[t] = x[2*W]
                gc[t] = d[t]-dc[t]
                xcc = xc[t]
                cost[t] = p[t]*(cc[t]+d[t]-dc[t])
            else: 
                w = T-t
                pw = np.r_[np.array(p[t]),pred_price[:w-1]]
                dw = d[t:t+w]
                dw = dw.reshape(w)
                x = self.solvelp(pw,dw,xcc)
                dc[t] = x[0]
                cc[t] = x[w]
                xc[t] = x[2*w]
                gc[t] = d[t]-dc[t]
                xcc = xc[t]
                cost[t] = p[t]*(cc[t]+d[t]-dc[t])
        return xc,dc,gc,cc,cost
    
    def mpc(self):
        n_interval = int(self.T/self.I); xc = 0
        cost_mpc = np.zeros(n_interval)
        clf = self.estimate_gmm(self.P[:self.N]); ex = 0;
        k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
        for i in range(k):
          ex = ex + por[i]*norm.expect(self.norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
        for n in range(1, n_interval+1):
            r_pred = self.pred_step(0,(n-1)*self.I,(n)*self.I,"Wind")
            r_pred = r_pred.astype(int)
            r_pred = r_pred.reshape(len(r_pred))
            
            a_real = self.D[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            
            d_real = (a_real - r_real)
            d_pred = (a_real - r_pred)
            
            for i in range(len(d_pred)):
                d_pred[i] = max(d_pred[i],0)
            
            d_real = d_real.astype(int); d_real = np.r_[np.array(0),d_real]
            d_pred = d_pred.astype(int); d_pred = np.r_[np.array(0),d_pred]
            p_real = np.r_[np.array(self.P[self.N+self.V+(n-1)*self.I-1]),p_real]
            
            x,d,va,vb,cost = self.Ampc_hat(d_pred,p_real,(n-1)*self.I,n*self.I,xc)
            cost_mpc[n-1],xc = self.Aour_demand(d,vb,d_real,p_real)
            
        cost_mpc_copy = cost_mpc.copy()
        for i in range(len(cost_mpc_copy)):
          cost_mpc[i] = sum(cost_mpc_copy[:i+1])
        return cost_mpc
      
    def isDecompose(self,at,x0):
        Acc_sum = 0
        Acc = np.zeros(len(at))
        for i in range(len(at)):
            Acc[i] = Acc_sum + at[i]
            Acc_sum = Acc_sum + at[i]
        AccB = Acc + self.B
        
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
                    sadwA[j] = i+1
                    i = i + 1
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
                        sadwA[j] = i + 1
                        k = k - 1
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

    def Athb(self, theta,BB,t,at,pt,xt0,tnz):
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
            vbt = min(max(BB-xt0,0),mu_c)
        else:
            dt =  min(at,mu_d,xt0)
            vat = at - dt
            vbt = 0
        xt = xt0+vbt-dt
        return xt,dt,vat,vbt
                
    def Athb_ld(self,theta,ts,tnz,abar,pt,t,xt0):
        if t == tnz:
            abart = abar
        else:
            abart = 0
        return self.Athb(theta,abar,t,abart,pt,xt0,tnz)
    
    def Athb_hat(self, ao,po,theta,xc):
        a = ao.copy() ; p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = 0;
        abar, ts, tnz = self.isDecompose(a,x0)
        xi = np.zeros((len(abar),len(a)))
        cost_shot = np.zeros((len(abar),len(a)))
        for t in range(len(a)):
            for i in range(len(abar)):
                if t>=ts[i] and t<=tnz[i]:
                    if t==ts[i]:
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
      
    def thb(self):
        n_interval = int(self.T/self.I); xc = 0
        cost_thb = np.zeros(n_interval)
        for n in range(1, n_interval+1):
            r_pred = self.pred_step(0,(n-1)*self.I,(n)*self.I,"Wind")
            r_pred = r_pred.astype(int)
            r_pred = r_pred.reshape(len(r_pred))
            
            a_real = self.D[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            
            d_real = (a_real - r_real)
            d_pred = (a_real - r_pred)
            
            for i in range(len(d_pred)):
                d_pred[i] = max(d_pred[i],0)
            
            d_real = d_real.astype(int); d_real = np.r_[np.array(0),d_real]
            d_pred = d_pred.astype(int); d_pred = np.r_[np.array(0),d_pred]
            p_real = np.r_[np.array(self.P[self.N+self.V+(n-1)*self.I-1]),p_real]
            
            p_train = self.P[:self.N]
            theta = math.sqrt(max(p_train[np.nonzero(p_train)])*min(p_train[np.nonzero(p_train)]))
            x,d,va,vb,cost_shot = self.Athb_hat(d_pred,p_real,theta,xc)
            cost_thb[n-1],xc = self.Aour_demand(d,vb,d_real,p_real)
            
        cost_thb_copy = cost_thb.copy()
        for i in range(len(cost_thb_copy)):
          cost_thb[i] = sum(cost_thb_copy[:i+1])
        return cost_thb
      
    def Aofl(self,ts,tnz,abar,p):
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
    
    def Aofl_hat(self,ao,po,xc):
        a = ao.copy() ; p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = xc; x[0] = 0
        abar, ts, tnz = self.isDecompose(a,x0)
        cost_shot = np.zeros(len(a))
        for i in range(len(abar)):
            xt,dt,vat,vbt = self.Aofl(ts[i],tnz[i],abar[i],p)
            for t in range(int(ts[i]),int(tnz[i]+1)):
                x[t] = x[t] + xt[t]
                d[t] = d[t] + dt[t]
                va[t] = va[t] + vat[t]
                vb[t] = vb[t] + vbt[t]
                a[t] = a[t] - dt[t] - vat[t]
                if vat[t]+vbt[t]>0:
                    cost_shot[t] = cost_shot[t] + p[t]*(vat[t]+vbt[t])
        return x,d,va,vb,cost_shot
    
    def ofl(self):
        n_interval = int(self.T/self.I); xc = 0
        cost_ofl = np.zeros(n_interval)
        for n in range(1, n_interval+1):
            r_pred = self.pred_step(0,(n-1)*self.I,(n)*self.I,"Wind")
            r_pred = r_pred.astype(int)
            r_pred = r_pred.reshape(len(r_pred))
            
            a_real = self.D[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            
            d_real = (a_real - r_real)
            d_pred = (a_real - r_pred)
            
            for i in range(len(d_pred)):
                d_pred[i] = max(d_pred[i],0)
            
            d_real = d_real.astype(int); d_real = np.r_[np.array(0),d_real]
            d_pred = d_pred.astype(int); d_pred = np.r_[np.array(0),d_pred]
            p_real = np.r_[np.array(self.P[self.N+self.V+(n-1)*self.I-1]),p_real]
            
            x,d,va,vb,cost_shot = self.Aofl_hat(d_pred,p_real,xc)
            cost_ofl[n-1],xc = self.Aour_demand(d,vb,d_real,p_real)
        cost_ofl_copy = cost_ofl.copy()
        for i in range(len(cost_ofl_copy)):
          cost_ofl[i] = sum(cost_ofl_copy[:i+1])
        return cost_ofl
    
    ###########################################################################
    def norm_pdfx(self,x):
        return x
    
    def estimate_gmm(self,X):
        X = X.reshape(-1,1)
        bic = []; lowest_bic = np.infty;
        n_components_range = range(1, 4)
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
    
    def sample_gaussian(self,mean, covar, covariance_type='diag', n_samples=1,
                        random_state=None):
        n_dim = len(mean)
        rand = np.random.randn(n_dim, n_samples)
        if n_samples == 1:
            rand.shape = (n_dim,)
        if covariance_type == 'spherical':
            rand *= np.sqrt(covar)
        elif covariance_type == 'diag':
            rand = np.dot(np.diag(np.sqrt(covar)), rand)
        else:
            s, U = linalg.eigh(covar)
            s.clip(0, out=s)        
            np.sqrt(s, out=s)
            U *= s
            rand = np.dot(U, rand)
        return (rand.T + mean).T
    
    def sample(self,clf, n_samples=1, random_state=None):
    
        weight_cdf = np.cumsum(clf.weights_)
    
        X = np.empty((n_samples, clf.means_.shape[1]))
        rand = np.random.rand(n_samples)
        comps = weight_cdf.searchsorted(rand)
        for comp in range(clf.n_components):
            comp_in_X = (comp == comps)
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                if clf.covariance_type == 'tied':
                    cv = clf.covars_
                elif clf.covariance_type == 'spherical':
                    cv = clf.covariances_[comp]
                else:
                    cv = clf.covars_[comp]
                X[comp_in_X] = self.sample_gaussian(
                    clf.means_[comp], cv, clf.covariance_type,
                    num_comp_in_X, random_state=random_state).T
        return X
    
    def Atheta_gmm(self,clf,T):
        k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
        t = int(T-2); theta = np.zeros(int(T))
        trunc = 0; re = 0;
        for i in range(k):
            trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
            re = re + por[i]*norm.expect(self.norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
        theta[int(T-2)] = re/trunc
        while(t>0):
            t = t-1; re1 = 0; re2 = 0; trunc = 0
            for i in range(k):
                trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
                re1 = re1 + por[i]*norm.expect(self.norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0,ub= theta[t+1])
                re2 = re2 + por[i]*theta[t+1] * (1-norm.cdf(theta[t+1],loc = mean[i], scale = np.sqrt(std[i])))
            theta[t] = (re1+re2)/trunc
        return theta
    
    def Aour_gmm(self,ao,po,clf,xc):
        a = ao.copy(); p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = xc; 
        abar, ts, tnz = self.isDecompose(a,x0)
        xi = np.zeros((len(abar),len(a)))
        theta = np.zeros((len(abar),len(a)))
        cost_shot = np.zeros((len(abar),len(a)))
        
        Theta = self.Atheta_gmm(clf,30)
        
        for t in range(len(a)):
            for i in range(len(abar)):
                if t>=ts[i] and t<=tnz[i]:
                    theta = Theta[30-int(tnz[i]-t)-1]
                    if t==ts[i]:
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
    
    def deta(self):
        n_interval = int(self.T/self.I); xc = 0
        cost_deta = np.zeros(n_interval)
        for n in range(1, n_interval+1):
            r_pred = self.pred_step(0,(n-1)*self.I,(n)*self.I,"Wind")
            r_pred = r_pred.astype(int)
            r_pred = r_pred.reshape(len(r_pred))
            
            a_real = self.D[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            
            d_real = (a_real - r_real)
            d_pred = (a_real - r_pred)
            
            for i in range(len(d_pred)):
                d_pred[i] = max(d_pred[i],0)
            
            d_real = d_real.astype(int); d_real = np.r_[np.array(0),d_real]
            d_pred = d_pred.astype(int); d_pred = np.r_[np.array(0),d_pred]
            p_real = np.r_[np.array(self.P[self.N+self.V+(n-1)*self.I-1]),p_real]
            
            clf = self.estimate_gmm(self.P[:self.N])
            x,d,va,vb,cost_shot = self.Aour_gmm(d_pred,p_real,clf,xc)
            cost_deta[n-1],xc = self.Aour_demand(d,vb,d_real,p_real)
        cost_deta_copy = cost_deta.copy()
        for i in range(len(cost_deta_copy)):
          cost_deta[i] = sum(cost_deta_copy[:i+1])
        return cost_deta
      
    def pred_step(self,t,TS,TD,MODEL_FLAG):
        
        if MODEL_FLAG == "Price":
          model = self.PM
          scaler = self.PS
          data = self.PSD
        else:
          model = self.WM
          scaler = self.WS
          data = self.WSD
        
        TOTAL_LEN = len(data[:,0])
        VALID_SAT = len(data[self.N+self.V+t+TS:,0])
        VALID_END = len(data[self.N+self.V+TD:,0])
        
        inputs = data[TOTAL_LEN - VALID_SAT - self.BD:
            TOTAL_LEN - VALID_END,:]
        
        pred = []; test_copy = inputs.copy()
        
        if self.LSTM == "Point":
            for i in range(self.BD,inputs.shape[0]):
                x_test = test_copy[i-self.BD:i,:]
                x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
                test_predict_v = model.predict(x_test)
                if test_predict_v[0][0]>1:
                    test_copy[i,0] = 1
                    pred.append(1)
                elif test_predict_v[0][0]<0:
                    test_copy[i,0] = 0
                    pred.append(0)
                else:
                    test_copy[i,0] = test_predict_v[0][0]
                    pred.append(test_predict_v[0][0])
        
        if self.LSTM == "S2S":
            for i in range(self.BD,inputs.shape[0]):
                if (i-self.BD)%10==0:
                    x_test = test_copy[i-self.BD:i,:]
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
                            
        if self.LSTM == "Vector":
            for i in range(self.BD,inputs.shape[0]):
                if (i-self.BD)%10==0:
                    x_test = test_copy[i-self.BD:i,:]
                    x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
                    test_predict_v = model.predict(x_test)
                    for j in range(10):
                        #print(test_predict_v[0][j][0])
                        if test_predict_v[0][j]>1:
                            test_copy[i,0] = 1
                            pred.append(1)
                        elif test_predict_v[0][j]<0:
                            test_copy[i,0] = 0
                            pred.append(0)
                        else:
                            test_copy[i,0] = test_predict_v[0][j]
                            pred.append(test_predict_v[0][j])
                            
        pred = np.array(pred)
        test_predict = scaler.inverse_transform(pred.reshape(-1, 1))
        
        return test_predict
        
    def Aour_pred_transfer(self,abar,ts,tnz,TS,TD):
        #data = scaler.inverse_transform(scaled_data)
        test_price = self.P[self.N+self.V+TS-1:self.N+self.V+TD]
        buy = np.zeros(len(abar))
        cost = np.zeros(len(abar))
        discharge = np.zeros(len(test_price))
        charge = np.zeros(len(test_price))
        
        for t in range(len(test_price)):
            try:
                prediction = self.pred_step(t,TS,TD,"Price")
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
    
    def Aour_demand(self,d_s,c_s,a,p):
        T = len(d_s)
        x = np.zeros(T)
        d = np.zeros(T)
        va = np.zeros(T)
        vb = np.zeros(T)
        cost = 0
        for t in range(T):
            if t == 0 :
                d[t] = min(a[t],0,d_s[t])
                va[t] = a[t] - d[t]
                vb[t] = min(self.B-0+d[t],c_s[t])
                x[t] = 0 +vb[t]-d[t]
                cost = cost + (va[t]+vb[t])*p[t]
            else:
                d[t] = min(a[t],x[t-1],d_s[t])
                va[t] = a[t] - d[t]
                vb[t] = min(self.B-x[t-1]+d[t],c_s[t])
                x[t] = x[t-1] +vb[t]-d[t]
                cost = cost + (va[t]+vb[t])*p[t]
        return cost,x[t]
    
    def leta(self):
        n_interval = int(self.T/self.I); xc = 0
        cost = np.zeros(n_interval)
        for n in range(1, n_interval+1):
            r_pred = self.pred_step(0,(n-1)*self.I,(n)*self.I,"Wind")
            r_pred = r_pred.astype(int)
            r_pred = r_pred.reshape(len(r_pred))
            
            a_real = self.D[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            
            d_real = (a_real - r_real)
            d_pred = (a_real - r_pred)
            
            for i in range(len(d_pred)):
                d_pred[i] = max(d_pred[i],0)
            
            d_real = d_real.astype(int); d_real = np.r_[np.array(0),d_real]
            d_pred = d_pred.astype(int); d_pred = np.r_[np.array(0),d_pred]
            p_real = np.r_[np.array(self.P[self.N+self.V+(n-1)*self.I-1]),p_real]
            
            abar, ts, tnz = self.isDecompose(d_pred,xc)
            d,vb,c = self.Aour_pred_transfer(abar,ts,tnz,(n-1)*self.I,(n)*self.I)
            cost[n-1],xc = self.Aour_demand(d,vb,d_real,p_real)
            
        cost_copy = cost.copy()
        for i in range(len(cost_copy)):
          cost[i] = sum(cost_copy[:i+1])
        return cost
      
    def nos(self):
        n_interval = int(self.T/self.I)
        cost = np.zeros(n_interval)
        for n in range(1, n_interval+1):
            
            a_real = self.D[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            d_real = (a_real - r_real)
            
            
            d_real = d_real.astype(int); d_real = np.r_[np.array(0),d_real]
            p_real = np.r_[np.array(self.P[self.N+self.V+(n-1)*self.I-1]),p_real]
            
            cost[n-1] = sum(d_real*p_real)
            
        cost_copy = cost.copy()
        for i in range(len(cost_copy)):
          cost[i] = sum(cost_copy[:i+1])
        return cost