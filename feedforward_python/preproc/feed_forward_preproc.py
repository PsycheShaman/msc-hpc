print("starting")

import glob

print("imported glob")

files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/unprocessed" + '/**/*.txt', recursive=True)

a = list(range(1,len(files)-1))

files_in_order = []
for i in a:
    files_in_order.append(files[i])

print("read files list")
    
from ast import literal_eval

print("imported literal_eval")

d = {}

print("concatenating files.............................................................")

for i in range(0,len(files_in_order)):
            print(files_in_order[i])
            di = open(files_in_order[i])
            di = di.read()
            if di == "}":
                continue
            else:
                di = di + "}"
                di = literal_eval(di)
                ki = list(di.keys())
                j = len(d)
                for k in ki:
                    j += 1000000000000000000000000000000123
                    di[j] = di.pop(k)
                    d.update(di)
            print(str(100*(i/len(files_in_order))))

k = d.keys()

pdgCode = []

layer0 = []

layer1 = []

layer2 = []

layer3 = []

layer4 = []

layer5 = []

print("extracting data from dictionaries................................................")

for i in k:
    pdgCode_i = d.get(i).get('pdgCode')    
    layer0_i = d.get(i).get('layer 0')
    layer1_i = d.get(i).get('layer 1')
    layer2_i = d.get(i).get('layer 2')
    layer3_i = d.get(i).get('layer 3')
    layer4_i = d.get(i).get('layer 4')
    layer5_i = d.get(i).get('layer 5')

    pdgCode.append(pdgCode_i)
    
    layer0.append(layer0_i)
    layer1.append(layer1_i)
    layer2.append(layer2_i)
    layer3.append(layer3_i)
    layer4.append(layer4_i)
    layer5.append(layer5_i)
    
    
electron = []

for i in pdgCode:
    if abs(i)==11:
        electron.append(1)
    else:
        electron.append(0)

import numpy as np

print("getting x and y values................................................")

for i in range(len(layer0)):
    if type(layer0[i])==type(None) or np.array(layer0[i]).shape==(17,0):
        continue
    else:
        x = np.array(layer0[i])
        x = np.sum(x,axis=0)
        y = np.array(electron[i])
        beg=i
        break
    
for i in range(beg+1,len(layer0)):
    if type(layer0[i])==type(None) or np.array(layer0[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer0[i])
        xi = np.sum(xi,axis=0)
        yi = electron[i]
        x = np.concatenate((x,xi))
        y = np.append(y,yi)
    if type(layer1[i])==type(None) or np.array(layer1[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer1[i])
        xi = np.sum(xi,axis=0)
        yi = electron[i]
        x = np.concatenate((x,xi))
        y = np.append(y,yi)
    if type(layer2[i])==type(None) or np.array(layer2[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer2[i])
        xi = np.sum(xi,axis=0)
        yi = electron[i]
        x = np.concatenate((x,xi))
        y = np.append(y,yi)
    if type(layer3[i])==type(None) or np.array(layer3[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer3[i])
        xi = np.sum(xi,axis=0)
        yi = electron[i]
        x = np.concatenate((x,xi))
        y = np.append(y,yi)
    if type(layer4[i])==type(None) or np.array(layer4[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer4[i])
        xi = np.sum(xi,axis=0)
        yi = electron[i]
        x = np.concatenate((x,xi))
        y = np.append(y,yi)
    if type(layer5[i])==type(None) or np.array(layer5[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer5[i])
        xi = np.sum(xi,axis=0)
        yi = electron[i]
        x = np.concatenate((x,xi))
        y = np.append(y,yi)
    print(str(100*i/len(layer0)))
        
x = np.array(x)
y = np.array(y)

x = np.reshape(x,(len(y),24))
x = x.astype('float32')

mu = np.mean(x)
x /= mu

import pickle
 
with open('/scratch/vljchr004/data/msc-thesis-data/x.pkl', 'wb') as x_file:
  pickle.dump(x, x_file)
  
with open('/scratch/vljchr004/msc-thesis-data/y.pkl', 'wb') as y_file:
  pickle.dump(y, y_file)

























