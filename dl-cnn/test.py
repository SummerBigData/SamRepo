import numpy as np
import random
import json
import argparse

from cnn_util import load_data
from softmax_util import predict
from nn_util import thetas_from_flat

parser = argparse.ArgumentParser(description='Neural Network backpropagation')
parser.add_argument('--name', dest='name', type=str, default='test', help='help ya dam self')
parser.add_argument('--shd_disp', dest='display', action='store_true', help='Whether or not to display the confidence histograms for every digit')
args = parser.parse_args()
name = args.name

X, y = load_data(train=False, conv=True, path='logs/'+args.name)
m = len(X)
k = len(y[0])

s = []
with open('logs/'+args.name+'/model.json') as f:
    s = json.loads(f.readline())['s']
sizes = [(s[i+1], s[i]+1) for i in range(len(s)-1)]
print sizes

theta_flat = np.genfromtxt('logs/'+name+'/weights.txt')
thetas = thetas_from_flat(theta_flat, sizes)

h, pred = predict(X, thetas, return_h=True)
actual = np.argmax(y, axis=1)
num_correct = np.sum(pred == actual)
print 'Model achieved %f percent accuracy on the test set.' % (num_correct/float(m))

corr_conf = []
wrong_conf = []
accs = []

for i in range(k):
    idxs = np.argwhere(actual == i)
    corr_conf.extend(h[idxs,i])

    num_pred = pred[idxs].flatten()
    wrong_idxs = np.argwhere(num_pred != i).flatten()
    wrong_conf.extend(np.max(h[idxs][wrong_idxs], axis=2).flatten())

    accs.append(np.sum(num_pred == i)/float(len(idxs)))
    print 'Accuracy for %d: %f' % (i, accs[-1])

    idx = np.random.choice(len(idxs))
    print 'n = %d: %f' % (i, h[idxs[idx]][0,i])
    
    print '***'

for i in range(len(corr_conf)):
    corr_conf[i] = corr_conf[i][0]

if args.display:
    from plotly.offline import plot
    import plotly.graph_objs as go

    data = [
        go.Histogram(
            x=corr_conf,
            cumulative=dict(enabled=True),
            opacity=0.75),
        go.Histogram(
            x=corr_conf,
            opacity=0.75
        )]
    layout = go.Layout(
        title='Confidence of actual digit in prediction',
        xaxis=dict(
            title='%% confidence',
            dtick=0.05),
        yaxis=dict(title='Count'),
        barmode='overlay'
    )
    plot({'data': data, 'layout': layout}, filename='pred_corr.html')

    data = [
        go.Histogram(
            x=wrong_conf,
            cumulative=dict(enabled=True),
            opacity=0.75),
        go.Histogram(
            x=wrong_conf,
            opacity=0.75
        )]
    layout = go.Layout(
        title='Confidence of incorrect digit in prediction',
        xaxis=dict(
            title='%% confidence',
            dtick=0.05),
        yaxis=dict(title='Count'),
        barmode='overlay'
    )
    plot({'data': data, 'layout': layout}, filename='pred_wrong.html')
    #plot({'data': data, 'layout': layout}, filename='pred.html')
