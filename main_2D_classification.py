# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:08:54 2022

@author: Daniel
"""
import sys
sys.path.append('./../python_ml_package')
import neuralNetwork as NN
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
import sklearn.datasets as sets
import tensorflow as tf


import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go



# parameters
moons_epochs = 100
moons_batchsize = 256
moons_alpha = 0.01
moons_beta1 = 0.8
moons_beta2 = 0.9
moons_lam = 0
moons_decayRate = 0.999
moons_layers = [2,100,100,2]


# -------------------- chosses Dataset --------------------
dataset = sets.make_moons()

colors =np.array(["#377eb8","#ff7f00"])

# -------------------- load and prepocess data --------------------
x = dataset[0]
y_raw = np.reshape(dataset[1],(-1,1))
y = NN.recodeY(y_raw)
x_steps = np.linspace(np.min(x[:,0]), np.max(x[:,0]))
y_steps = np.linspace(np.min(x[:,1]), np.max(x[:,1]))
x_grid = np.meshgrid(x_steps, y_steps)
x_grid_unrolled = np.array([x_grid[0].flatten(),x_grid[1].flatten()]).T


# -------------------- create and train model --------------------
model = NN.NeuralNetwork(numberNodes = moons_layers,
                         activation = 4,
                         classification = True)
Js, record_y = model.gradientDescent(x, y, moons_lam, moons_epochs, moons_alpha, 
                      beta1 = moons_beta1, beta2 = moons_beta2, 
                      batchsize = moons_batchsize, 
                      decayRate = moons_decayRate,
                      number_records = 25,
                      record_x = x_grid_unrolled)
y_grid_unrolled_hat = model.predict(x_grid_unrolled)
model.accuracy(x, y_raw)

#y_show = 
#i_frame = 2
#h = record_y[i_frame,:,:]
#y_show = np.where(h == np.amax(h,axis=1,keepdims=True))[1]
# -------------------- plot result --------------------
recorded_frames = []
for yi in record_y:
    f = np.where(yi == np.amax(yi,axis=1,keepdims=True))[1]\
        .reshape(x_grid[0].shape)
    recorded_frames.append(go.Frame(data=[go.Contour(x = x_steps,
                                                     y = y_steps,
                                                     z = f,
                                                     colorscale = [[0, "rgba(55, 126, 184, 0.1)"],
                                                                   [1, "rgba(255, 127, 0, 0.1)"]])]))

moon_trace = go.Scatter(x=x[:,0], y=x[:,1],mode='markers',
                        marker=dict(size = 15, color=colors[y_raw.flatten()]))
fig = go.Figure(
    data=[go.Contour(), go.Scatter()],
    layout=go.Layout(
        xaxis=dict(range=[np.min(x_steps), np.max(x_steps)], autorange=False),
        yaxis=dict(range=[np.min(y_steps), np.max(y_steps)], autorange=False),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
        ),
    frames = recorded_frames
    )
fig.add_trace(moon_trace)
fig.show()
plot(fig, auto_open=True)


#df = px.data.gapminder()
#
#a = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
#           size="pop", color="continent", hover_name="country",
#           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
#fig = go.Figure(a)
#plot(fig, auto_open=True)

#a = px.Contour(x=x_steps,y = y_steps,z=0, animation_frame="year", 
#               range_x=[np.min(x_steps), np.max(x_steps)], 
#               range_y=[np.min(y_steps), np.max(y_steps)])

#fig = go.Figure(a)
#plot(fig, auto_open=True)

'''
fig, ax = plt.subplots()
CS = ax.contourf(x_grid[0],x_grid[1], 
                y_show.reshape(x_grid[0].shape), 
                levels = [0,0.5,1],
                colors = colors,
                alpha = 0.3)
CS.set
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Decicion boundary')
plt.scatter(x[:,0],x[:,1],color=colors[y_raw.flatten()])'''