import numpy as np
import pandas as pd

from plotly.offline import plot
import plotly.graph_objs as go

import os

dirs = os.listdir('logs')
dirs = list(filter(lambda f: os.path.isdir('logs/'+f), dirs))

data = dict()

for name in dirs:
    if not os.path.isfile('logs/'+name+'/cost_and_acc.csv'):
        print name + ' does not have cost.csv'
        continue
    df = pd.read_csv('logs/'+name+'/cost_and_acc.csv')
    if len(df['Cost'].as_matrix()) == 0:
        continue
    data[name] = (df['Cost'].as_matrix(), df[' Accuracy'].as_matrix())

for name, (cost, acc) in data.items():
    if len(cost) < 3:
        continue
    data = [
        go.Scatter(x=list(range(len(cost))),
            y=cost,
            mode='lines+markers',
            name='Cost'),
        go.Scatter(x=list(range(len(acc))),
            y=acc,
            mode='lines+markers',
            name='Accuracy')
    ]
    layout = go.Layout(
        title=name,
        xaxis=dict(title='Iteration * 20'),
        yaxis=dict(title='Metric Value')
    )
    plot({'data': data, 'layout': layout}, filename='plots/'+name+'_metrics.html')

"""
cost_traces = []
acc_traces = []
for name, (cost, acc) in data.items():
    cost_traces.append(
        go.Scatter(
            x=list(range(len(cost))),
            y=cost,
            mode='lines+markers',
            name=name))
    acc_traces.append(
        go.Scatter(
            x=list(range(len(acc))),
            y=acc,
            mode='lines+markers',
            name=name))

layout = go.Layout(
    title='Cost over Time',
    xaxis=dict(title='Iteration * 20'),
    yaxis=dict(title='Cost'))

plot({'data': cost_traces, 'layout': layout}, filename='costs.html')

layout = go.Layout(
    title='Accuracy over Time',
    xaxis=dict(title='Iteration * 20'),
    yaxis=dict(title='Accuracy'))

plot({'data': acc_traces, 'layout': layout}, filename='accs.html')
"""