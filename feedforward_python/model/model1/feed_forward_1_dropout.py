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
    
from tensorflow.keras.utils import to_categorical

#y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

import tensorflow

from tensorflow import keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


num_classes = 2
epochs = 100

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    
model1_dropout_0_5 = Sequential([
    Dense(256, input_shape=(24,)),
    Activation('relu'),
    Dropout(0.5),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(2),
    Activation('softmax')
])

model1_dropout_0_5.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model1_dropout_0_5.fit(x_train, y_train,
              #batch_size=batch_size,
              epochs=epochs,
              validation_split=0.15,
              shuffle=True,
              verbose=2)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_dropout_0_5_history.png', bbox_inches='tight')

model1_dropout_0_5.probs = model1_dropout_0_5.predict_proba(x_test)

import numpy as np
np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1__dropout_0_5_results.csv", np.array(model1_dropout_0_5.probs), fmt="%s")

model1_dropout_0_5.save('/home/vljchr004/msc-hpc/feedforward_python/feed_forward_1__dropout_0_5.h5')  # creates a HDF5 file 'my_model.h5'
del model1_dropout_0_5 

model1_dropout_0_8 = Sequential([
    Dense(256, input_shape=(24,)),
    Activation('relu'),
    Dropout(0.8),
    Dense(128),
    Activation('relu'),
    Dropout(0.8),
    Dense(128),
    Activation('relu'),
    Dropout(0.8),
    Dense(64),
    Activation('relu'),
    Dropout(0.8),
    Dense(2),
    Activation('softmax')
])

model1_dropout_0_8.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model1_dropout_0_8.fit(x_train, y_train,
              #batch_size=batch_size,
              epochs=epochs,
              validation_split=0.15,
              shuffle=True,
              verbose=2)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_dropout_0_8_history.png', bbox_inches='tight')

model1_dropout_0_8.probs = model1_dropout_0_8.predict_proba(x_test)

import numpy as np
np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1__dropout_0_8_results.csv", np.array(model1_dropout_0_8.probs), fmt="%s")

model1_dropout_0_8.save('/home/vljchr004/msc-hpc/feedforward_python/feed_forward_1__dropout_0_8.h5')  # creates a HDF5 file 'my_model.h5'
del model1_dropout_0_8

model1_dropout_0_8_0_5 = Sequential([
    Dense(256, input_shape=(24,)),
    Activation('relu'),
    Dropout(0.8),
    Dense(128),
    Activation('relu'),
    Dropout(0.8),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(2),
    Activation('softmax')
])

model1_dropout_0_8_0_5.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model1_dropout_0_8_0_5.fit(x_train, y_train,
              #batch_size=batch_size,
              epochs=epochs,
              validation_split=0.15,
              shuffle=True,
              verbose=2)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_dropout_0_8_0_5_history.png', bbox_inches='tight')

model1_dropout_0_8_0_5.probs = model1_dropout_0_8_0_5.predict_proba(x_test)

import numpy as np
np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1__dropout_0_8_0_5_results.csv", np.array(model1_dropout_0_8_0_5.probs), fmt="%s")

model1_dropout_0_8_0_5.save('/home/vljchr004/msc-hpc/feedforward_python/feed_forward_1__dropout_0_8_0_5.h5')  # creates a HDF5 file 'my_model.h5'
del model1_dropout_0_8_0_5

