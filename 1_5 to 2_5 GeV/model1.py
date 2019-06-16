# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 18:47:05 2019

@author: gerhard
"""

import glob

import numpy as np

P_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/P_*.pkl", recursive=True)

#from numpy import genfromtxt
#
#P = genfromtxt(P_files[0], delimiter=',')
#
#for i in P_files[1:]:
#    Pi = genfromtxt(i, delimiter=',')
#    P = np.concatenate((P,Pi),axis=None)


x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.pkl")
y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.pkl")

x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\y_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)

with open(P_files[0], 'rb') as P_file0:
   P = pickle.load(P_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
    with open(i,'rb') as x_file:
        print(i)
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)
        
x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.npy")
y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.npy")
       
print("recursively adding x numpys........................................................................................")

for i in x_files[0:]:
    with open(i,'rb') as x_file:
        print(i)
        xi = np.load(x_file)
        x = np.concatenate((x,xi),axis=0)

print("recursively adding y numpys........................................................................................")

for i in y_files[0:]:
    with open(i,'rb') as y_file:
        yi = np.load(y_file)
        y = np.concatenate((y,yi),axis=None)
        
nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)
#P = np.delete(P,zeros)

x.shape = (x.shape[0],x.shape[1],x.shape[2],1)

#GeV_range2 = np.where(P>=1.8 and P<=2.2)
#
#x = x[GeV_range2,:,:,:]
#y = y[GeV_range2]


electrons = np.where(y==1)

electrons = electrons[0]

pions = np.where(y==0)

pions = pions[0]

pions = pions[0:electrons.shape[0]]

x_1 = x[electrons,:,:,:]
x_2 = x[pions,:,:,:]

x = np.vstack((x_1,x_2))

y_1 = y[electrons]
y_2 = y[pions]

y = np.concatenate((y_1,y_2),axis=None)

ma = np.max(x)

x = x/ma

ma = np.amax(x,axis=2)

x = np.divide(x,ma)

#check the division above before running!!!!!!!!!!!1

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



import tensorflow

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM











