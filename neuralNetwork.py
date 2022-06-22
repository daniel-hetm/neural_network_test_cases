# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:50:58 2022

@author: Daniel
"""
import numpy as np
import scipy.optimize as op
import activationFunctions as af
import copy

class NeuralNetwork:
    theta = []
    b = []
    iteration = 0
    def __init__(self, numberNodes = None, weight_file = False, 
                 activation = 4, classification = True):
        self.actFunc, self.gradFunc = af.giveFunctions(activation)
        self.classification = classification
        if classification:
            self.outFunc, self.outGrad = af.giveFunctions(1)
        else:
            self.outFunc, self.outGrad = af.giveFunctions(0)
        if weight_file:
            self.loadParams()
        else:
            self.createParams(numberNodes)
        
    def createParams(self, numberNodes):
        for i in range(1, len(numberNodes)):
            e = np.sqrt(6) / np.sqrt(numberNodes[i-1] + numberNodes[i])
            print('Epsilon for Theta ', i, ' ', e)
            self.theta.append(2 * e * (np.random.rand(numberNodes[i-1], 
                                        numberNodes[i])) - e)
            self.b.append(np.zeros((1, numberNodes[i])))

    def h(self, X, ts = None, b = None):
        x = X.copy()
        if ts is None:
            ts = self.theta
            b = self.b
        for ti,bi in zip(ts[:-1], b[:-1]):
            x = self.actFunc(x.dot(ti) + bi)
        x = self.outFunc(x.dot(ts[-1]) + b[-1])
        return x
        
    
    def predict(self, x):
        h = self.h(x)
        if self.classification:
            return np.where(h == np.amax(h,axis=1,keepdims=True))[1]
        return h
    
    
    def costGrad(self, params, X, y, lam, batchsize):
        # -------------------- initialize params/data --------------------
        if batchsize == 0:
            x = X.copy()
        else:
            m = np.size(y,0)
            batch = np.random.choice(m, batchsize, False)
            x = X[batch, :]
            y = y[batch, :]
        m = np.size(y,0)
        if np.any(params) == None:
            transformParams = False
            theta, b = self.theta, self.b
        else:
            transformParams = True
            theta, b = self.roleUp(params)

        self.iteration += 1
        
        a = []
        dz = []
        z = []
        D = []

        # -------------------- forward propagation --------------------
        for i,(t,bi) in enumerate(zip(theta, b)):
            a.append(x)
            x = x.dot(t)+bi
            z.append(x)
            if i == (len(b)-1):
                x = self.outFunc(x)
            else:
                x = self.actFunc(x)
            dz.append(x)
        
        # -------------------- calculate cost --------------------
        h = x
        if self.classification:
            J = np.sum(np.multiply(-y,np.log(h)) - 
                   np.multiply((1-y), np.log(1-h))) / m
        else:
            J = np.sum(np.square(h-y))/m
        for t in theta:
            J += lam / (2*m) * np.sum(np.square(t))
        if transformParams:
            print('Cost (', self.iteration, '):', J)
        
        
        # -------------------- back propagation --------------------
        
        dw = copy.deepcopy(theta)
        db = copy.deepcopy(b)
        for i in reversed(range(0,len(theta))):
            if i == len(theta) - 1:
                dz[-1] -= y #derivative is in here!
            else:
                dz[i] = np.multiply(dz[i+1].dot(theta[i+1].T), 
                                     self.gradFunc(z[i]))
            dw[i] = np.dot(a[i].T,dz[i]) / m
            dw[i] += lam/m * theta[i]
            db[i] = np.sum(dz[i], axis = 0, keepdims = True) / m
            db[i] += lam/m * b[i]

        
        
        # -------------------- return results --------------------
        if transformParams:
            for dwi,dbi in zip(dw,db):
                D.extend(dwi.flatten())
                D.extend(dbi.flatten())
            return J,D
        else:
            return J,dw,db
    
    def train(self, x, y, lam, iterations, batchsize = 0):
        params = self.unrole()
        def decoratedCost(params):
            return (self.costGrad(params, x, y, lam, batchsize))
        result = op.minimize(decoratedCost, params, method='CG', 
                             jac=True, 
                             options = {'maxiter' : iterations})
        params = result.x
        self.theta, self.b = self.roleUp(params)
        self.saveParams()
    
    def gradientDescent(self, x, y, lam, max_iterations, alpha, batchsize = 0, 
                        beta1 = 0.9, beta2 = 0.999, eps = 10**-8, decayRate = 0,
                        number_records = 0, record_x = None):
        vdw,vdb,sdw,sdb = [],[],[],[]
        m = np.shape(y)[0]
        if batchsize == 0:
            batchsize = m
        for dwi,dbi in zip(self.theta,self.b):
            vdw.append(np.zeros(dwi.shape))
            vdb.append(np.zeros(dbi.shape))
            sdw.append(np.zeros(dwi.shape))
            sdb.append(np.zeros(dbi.shape))
        
        
        epochs_num = np.ceil(m/batchsize).astype(int)
        Js = np.zeros((max_iterations))
        if number_records > 0:
            record_y = np.zeros((number_records,
                                 record_x.shape[0],y.shape[1]))
            record_step = np.floor(max_iterations/number_records).astype(np.int)
        else:
            record_y = None
            record_step = None
        for i in range(max_iterations):
            ind_perm_all = np.random.permutation(m)
            Js_temp = 0
            for ei in range(epochs_num):
                if ei == epochs_num - 1:
                    ind_perm_epoch = ind_perm_all[ei*batchsize:]
                else:
                    ind_perm_epoch = ind_perm_all[ei*batchsize:(ei+1)*batchsize]
                J,dw,db = self.costGrad(None, 
                            x[ind_perm_epoch,:], y[ind_perm_epoch,:], lam, 0)
                Js_temp += J
                iteration = i * epochs_num + ei +1
                for l, (dwi,dbi) in enumerate(zip(dw,db)):
                    vdw[l] = (beta1 * vdw[l] + (1 - beta1) * dwi) \
                        / (1 - np.power(beta1,iteration))
                    vdb[l] = beta1 * vdb[l] + (1 - beta1) * dbi \
                        / (1 - np.power(beta1,iteration))
                    sdw[l] = (beta2 * sdw[l] + (1 - beta2) * np.square(dwi)) \
                        / (1 - beta2**iteration)
                    sdb[l] = (beta2 * sdb[l] + (1 - beta2) * np.square(dbi)) \
                        / (1 - beta2**iteration)
                    
                    self.theta[l] = self.theta[l] - alpha * vdw[l] \
                        / (np.sqrt(sdw[l]) + eps)
                    self.b[l] = self.b[l] - alpha * vdb[l] \
                        / (np.sqrt(sdb[l]) + eps)
            Js_temp /= epochs_num
            Js[i] = Js_temp
            # -------------------- reduce learning rate
            if doDecay(i,Js,decayRate):
                alpha *= decayRate
            print('Cost (', i, '):', round(Js_temp,8), 
                  ' Alpha: ', round(alpha,8))
            # ---------- record intermediate results ----------
            if number_records > 0:
                if i % record_step == 0:
                    i_record = (i / record_step).astype(np.int)
                    record_y[i_record,:,:] = self.h(record_x)
        self.saveParams()
        return Js, record_y
    
    def unrole(self, theta = None, b = None):
        if theta is None:
            theta = self.theta
            b = self.b
        params = []
        for t, bi in zip(theta, b):
            params.extend(t.flatten())
            params.extend(bi.flatten())
        return params
    
    def roleUp(self, params):
        pos = 0
        theta = []
        b = []
        for ti in self.theta:
            ti_shape = np.shape(ti)
            ti_size = ti_shape[0] * ti_shape[1]
            theta.append(np.reshape(params[pos:pos+ti_size], ti_shape))
            pos += ti_size 
            b.append(np.reshape(params[pos:pos+ti_shape[1]], (1, -1)))
            pos += ti_shape[1]
        return theta, b
    
    def saveParams(self):
        with open('weights.npz', 'wb') as f:
            np.savez(f,self.theta)
        b = []
        for bi in self.b:
            b.append(bi.flatten())
        with open('bias.npz', 'wb') as f:
            np.savez(f,b)
    
    def loadParams(self):
        self.theta = []
        self.b = []
        with open('weights.npz', 'rb') as f:
            temp = np.load(f, allow_pickle=True)['arr_0']
            for w in temp:
                self.theta.append(w)
        with open('bias.npz', 'rb') as f:
            temp = np.load(f, allow_pickle=True)['arr_0']
            for bi in temp:
                self.b.append(bi.reshape((1, -1)))
    
    def accuracy(self, x, y):
        h = self.predict(x)
        accurancy = sum(h == y.flatten())/np.size(y) * 100
        print('Accuracy: ', accurancy)
        return accurancy

def recodeY(y):
    n_classes = np.max(y) + 1
    m = np.size(y)
    y_recoded = np.zeros((m, n_classes))
    for i in range(n_classes):
        y_recoded[:,i] = (y == i).flatten()
    return y_recoded

def vecToMatrix(v):
    return np.reshape(v, (-1, 1))

def doDecay(i,Js,decayRate):
    if decayRate > 0:
        if i > 0:
            if Js[i] > Js[i-1]:
                return True
    return False



