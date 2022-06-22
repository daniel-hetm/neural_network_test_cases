# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:37:11 2022

@author: Daniel
"""

import sys
sys.path.append('./../python_ml_package')
import neuralNetwork as NN
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
import sklearn.datasets as sets

import tensorflow as tf


do_my_model = 1
do_sklearn = 0
do_tf_beginner = 0
do_tf_advanced = 0 # not working yet

# params
sin_epochs = 100
sin_batchsize = 256
sin_alpha = 0.01
sin_beta1 = 0.8
sin_beta2 = 0.9
sin_lam = 0
sin_decayRate = 0.999
sin_layers = [1,100,100,1]



# -------------------- initialize data --------------------
x = np.linspace(0, 1,1000).reshape((-1,1))
y = np.sin(4*np.pi*x)


if do_my_model:
    model = NN.NeuralNetwork(numberNodes = sin_layers,
                             activation = 4,
                             classification = False)
    model.gradientDescent(x, y, sin_lam, sin_epochs, sin_alpha, 
                          beta1 = sin_beta1, beta2 = sin_beta2, 
                          batchsize = sin_batchsize, 
                          decayRate = sin_decayRate)
    #model.train(x, y, 1, epochs)
    y_hat = model.predict(x)



# ----- sklearn training -----
if do_sklearn:
    clf = MLPRegressor(solver='lbfgs', alpha=1e-5, 
                        hidden_layer_sizes=(100, 100), random_state=1)
    clf.fit(x, y)
    y_hat2 = clf.predict(x)


# ----- tensorflow training -----
if do_tf_beginner:
    print("TensorFlow version:", tf.__version__)
    tf_model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(1),
      tf.keras.layers.Dense(100, activation='relu'),
      tf.keras.layers.Dense(100, activation='relu'),
      tf.keras.layers.Dense(1)
    ])
    loss_tf = tf.keras.losses.MeanSquaredError(
        name='mean_squared_error'
    )
    tf_model.compile(optimizer='adam',
                  loss=loss_tf,
                  metrics=['accuracy'])
    tf_model.fit(x, y, epochs=sin_epochs)
    tf_model.evaluate(x, y, verbose=2)
    y_hat_tf = tf_model(x)


if do_tf_advanced:
    relu = tf.keras.activations.relu
    x_tf = tf.Variable(x.T,dtype=tf.double)
    y_tf = tf.Variable(x.T,dtype=tf.double)
    initializer = tf.keras.initializers.HeNormal()
    w1 = tf.Variable(tf.cast(initializer(shape = (100,1)),dtype=tf.double))
    w2 = tf.Variable(tf.cast(initializer(shape = (100,100)),dtype=tf.double))
    w3 = tf.Variable(tf.cast(initializer(shape = (1,100)),dtype=tf.double))
    b1 = tf.Variable(tf.zeros((100,1),dtype=tf.double))
    b2 = tf.Variable(tf.zeros((100,1),dtype=tf.double))
    b3 = tf.Variable(tf.zeros(1,dtype=tf.double))
    opt = tf.keras.optimizers.Adam(0.01)
    trainable_variables = [w1,w2,w3,b1,b2,b3]
    def h():
        return w3@relu(w2@relu(w1@x_tf+b1)+b2)+b3
    def cost():
        return tf.reduce_mean(tf.square(tf.subtract(y_hat,y)))
    for i in range(1):
        with tf.GradientTape() as t:
            y_hat = h()
            c = cost()
        gradients = t.gradient(cost(), trainable_variables)
        step_count = opt.minimize(cost(), 
                    var_list=trainable_variables,
                    grad_loss=gradients,
                    tape = t)
    y_hat_tf2 = h()

plt.figure()
plt.plot(x,y)
if do_my_model:
    plt.plot(x,y_hat)
if do_sklearn:
    plt.plot(x,y_hat2)
if do_tf_beginner:
    plt.plot(x,y_hat_tf)
if do_tf_advanced:
    plt.plot(x,tf.transpose(y_hat_tf2))