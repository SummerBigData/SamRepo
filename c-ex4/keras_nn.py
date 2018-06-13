from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from nn_util import load_data

model = Sequential([
    Dense(50, input_shape=(784,)),
    Activation('sigmoid'),
    Dropout(0.5),
    Dense(25),
    Activation('sigmoid'),
    #Dropout(0.25),
    Dense(10),
    Activation('sigmoid')
])
model.compile(optimizer='rmsprop'
            , loss='categorical_crossentropy'
            , metrics=['accuracy'])

X, y = load_data(all_data=True, num_sample=60000)
X_test, y_test = load_data(all_data=True, train=False)

model.fit(X, y, epochs=25, batch_size=32, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, batch_size=32)
print 'FINAL SCORE: ', score[1]

w1 = model.layers[0].get_weights()
w2 = model.layers[3].get_weights()
w3 = model.layers[5].get_weights()

import numpy as np
theta1 = np.hstack((w1[0].T, w1[1].reshape((len(w1[1]), 1))))
theta2 = np.hstack((w2[0].T, w2[1].reshape((len(w2[1]), 1))))
theta3 = np.hstack((w3[0].T, w3[1].reshape((len(w3[1]), 1))))

import os
if not os.path.isdir('logs/keras'):
    os.mkdir('logs/keras')

tflat = np.array([])
for t in [theta1, theta2, theta3]:
    tflat = np.append(tflat, t.flatten())

np.savetxt('logs/keras/weights.txt', tflat)

import json

s = [784, 50, 25, 10]
with open('logs/keras/model.json', 'w') as f:
	f.write(json.dumps({'s': s}))