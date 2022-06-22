# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:30:37 2022

@author: Daniel
"""

import numpy as np

def giveFunctions(f):
    if f == 0:
        return lin, linGradient
    elif f == 1:
        return sigmoid, sigmoidGradient
    elif f == 2:
        return tanh, tanhGradient
    elif f == 3:
        return relu, reluGradient
    elif f == 4:
        return leakyRelu, leakyReluGradient


def lin(z):
    return z

def linGradient(z):
    return 1

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidGradient(z):
    g = sigmoid(z)
    return np.multiply(g, 1-g)

def tanh(z):
    return np.tanh(z)

def tanhGradient(z):
    a = np.tanh(z)
    return (1 - np.square(a))

def relu(z):
    g = z.copy()
    g[g<0] = 0
    return g

def reluGradient(z):
    g = np.ones(z.shape)
    g[z<0] = 0
    return g

def leakyRelu(z):
    g = z.copy()
    g[z<0] = 0.01*z[z<0]
    return g

def leakyReluGradient(z):
    g = np.ones(z.shape)
    g[z<0] = 0.01
    return g