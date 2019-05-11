# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:47:42 2019

@author: gerhard
"""

import pickle
 
with open('/scratch/vljchr004/data/msc-thesis-data/x.pkl', 'rb') as x_file:
    x = pickle.load(x_file)

with open('/scratch/vljchr004/data/msc-thesis-data/y.pkl', 'rb') as y_file:
    y = pickle.load(y_file)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import tensorflow

from tensorflow import keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


num_classes = 2
epochs = 1000

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
    
model1 = Sequential([
    Dense(512, input_shape=(24,)),
    Activation('relu'),
    Dense(512),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(128),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(2),
    Activation('softmax')
])

model1.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model1.fit(x_train, y_train,
              #batch_size=batch_size,
              epochs=epochs,
              validation_split=0.15,
              shuffle=True)

model1.probs = model1.predict_proba(x_test)