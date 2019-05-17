print("==============================================================================================")

#import argparse
#
#parser = argparse.ArgumentParser()
#parser.add_argument("run", help="enter the specific run you need to process",type=str)
#args = parser.parse_args()
#
#run = str(args.run)

#run = 'testDict'
#
#print("starting........................................................................................")
#
#import glob
#
#print("imported glob........................................................................................")
#
#files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-hpc\\" + run + '\\**\\*.txt', recursive=True)
#
#a = list(range(1,len(files)-1))

import os

p = os.getcwd()+'\\Documents\\msc-hpc'

os.chdir(p)

p = os.join

files_in_order = ['testDict.txt','testDict.txt']

print("read files list........................................................................................")

from ast import literal_eval

def file_reader1(i):
    di = open(i)
    di = di.read()
    if di == "}":
        pass
    else:
        di = di + "}"
        di = literal_eval(di)
        ki = list(di.keys())
        pdgCode = [di.get(k).get('pdgCode') for k in ki]
        return(pdgCode)
        
def file_reader2(i,l):
    di = open(i)
    di = di.read()
    if di == "}":
        pass
    else:
        di = di + "}"
        di = literal_eval(di)
        ki = list(di.keys())
        layer = [di.get(k).get(l) for k in ki]
#        y = [i for i in y if i is not None]
        return(layer)

print("pdg........................................................................................")
        
pdgCode = [file_reader1(i) for i in files_in_order]

import numpy as np

pdgCode = np.concatenate(pdgCode).ravel()

print("layer 0........................................................................................")

##################################

layer0 = [file_reader2(i,"layer 0") for i in files_in_order]

#layer0 = np.array(layer0)

layer0 = np.concatenate(layer0,axis=None)

empties = np.where([np.array(i).shape!=(17,24) for i in layer0])

layer0 = np.delete(layer0, empties)

#layer0 = np.concatenate(layer0,axis=None)

layer0 = np.stack(layer0)

pdgCode = np.delete(pdgCode, empties)




#####################################

print("layer 1........................................................................................")

layer1 = [file_reader2(i,"layer 1") for i in files_in_order]


print("layer 2........................................................................................")

layer2 = [file_reader2(i,"layer 2") for i in files_in_order]


print("layer 3........................................................................................")

layer3 = [file_reader2(i,"layer 3") for i in files_in_order]


print("layer 4........................................................................................")

layer4 = [file_reader2(i,"layer 4") for i in files_in_order]


print("layer 5........................................................................................")

layer5 = [file_reader2(i,"layer 5") for i in files_in_order]


print("mapped out files to useful elements....................................................................")

import numpy as np

def pdg_code_to_elec(i):
    if np.abs(i)==11:
        return(1)
    else:
        return(0)
        
electron = [pdg_code_to_elec(i) for i in pdgCode]

print("mapped out electrons....................................................................")

print(electron)

def x_0_getter(i):
    import numpy as np

    layer0 = i
    if type(layer0)==type(None) or np.array(layer0).shape==(17,0):
        pass
    else:
        x0 = np.array(layer0)
        x0 = [i for i in x0 if i is not None]
        x0 = [i for i in x0 if i is not []]
        x0=np.array(x0)
        x0.shape = (int(len(x)/17),24)
        x0 = np.sum(x0,axis=0)

    if 'x0' in locals():
        return(x0)
#

print("get x&y from layers........................................................................................")

def y_0_getter(electron,i):
    import numpy as np

    layer0 = i
    if type(layer0)==type(None) or np.array(layer0).shape==(17,0):
        pass
    else:
        y0 = np.array(electron)

    if 'y0' in locals():
        return(y0)

print("layer 0........................................................................................")

x0 = [x_0_getter(i) for i in layer0]
y0 = [y_0_getter(electron,i) for i in layer0]

print("x0........................................................................................")

print(x0)

print("y0........................................................................................")

print(y0)

print("layer 1........................................................................................")

x1 = [x_0_getter(i) for i in layer1]
y1 = [y_0_getter(electron,i) for i in layer1]

print("layer 2........................................................................................")

x2 = [x_0_getter(i) for i in layer2]
y2 = [y_0_getter(electron,i) for i in layer2]

print("layer 3........................................................................................")

x3 = [x_0_getter(i) for i in layer3]
y3 = [y_0_getter(electron,i) for i in layer3]

print("layer 4........................................................................................")

x4 = [x_0_getter(i) for i in layer4]
y4 = [y_0_getter(electron,i) for i in layer4]

print("layer 5........................................................................................")

x5 = [x_0_getter(i) for i in layer5]
y5 = [y_0_getter(electron,i) for i in layer5]

print("concatenating........................................................................................")

x = np.concatenate((x0,x1,x2,x3,x4,x5),axis=None)

x = x.flatten()

x = [i for i in x if i is not None]

x = np.concatenate(x).ravel()

y = np.concatenate((y0,y1,y2,y3,y4,y5),axis=None)

y = y.flatten()

y = [i for i in y if i is not None]

y = np.concatenate(y).ravel()

print("reshape x and y........................................................................................")

import numpy as np

x = np.reshape(x,(len(y),24))
x = x.astype('float32')

mu = np.mean(x)
x /= mu

print("x.......................................................................................................")

print(x)

print("y.......................................................................................................")

print(y)

print("pickling files........................................................................................")

import pickle

with open('/scratch/vljchr004/data/msc-thesis-data/x_' + run + '.pkl', 'wb') as x_file:
  pickle.dump(x, x_file)

with open('/scratch/vljchr004/data/msc-thesis-data/y_' + run + '.pkl', 'wb') as y_file:
  pickle.dump(y, y_file)


print("done.........................................................................................")

print("==============================================================================================")
