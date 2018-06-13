import numpy as np
import random
import json
import argparse

from nn_util import *

parser = argparse.ArgumentParser(description='Neural Network backpropagation')
parser.add_argument('--all_data', dest='use_all_data', action='store_true', help='Whether or not to use to full dataset')
parser.add_argument('--shd_split', dest='split', action='store_true', help='Whether or not to split the dataset into training and testing sets')
parser.add_argument('--name', dest='name', type=str, default='nn', help='help ya dam self')
parser.add_argument('--shd_disp', dest='display', action='store_true', help='Whether or not to display the confidence histograms for every digit')
args = parser.parse_args()
name = args.name

if args.split:
	_, _, X, y = load_data(all_data=args.use_all_data, train=False, split=True)
else:
	X, y = load_data(all_data=args.use_all_data, train=False)
m, n = X.shape[0], X.shape[1]
k = 10

s = []
with open('logs/'+name+'/model.json') as f:
	s = json.loads(f.readline())['s']
L = len(s)

sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
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
