 
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

d = {}

for i in range(0,len(files_in_order)):
#practice run on 5 files:
#for i in range(0,5):
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

print("layer and pdg")

print("get pdg")

def pdg_getter(d,i):
    
    print(str(i))
    
    pdgCode_i = d.get(i).get('pdgCode')    
    
    return(pdgCode_i)

import multiprocessing as mp

k = d.keys() 

print("get pdg")

pool = mp.Pool(mp.cpu_count())

pdgCode = [pool.apply(pdg_getter, args=(d,i)) for i in k]

pool.close()

electron = []

print("electron one-hot encoding")

import numpy as np

for i in pdgCode:
    if np.abs(i)==11:
        electron.append(1)
    else:
        electron.append(0)
        
print("get layer 0")
        
def layer0_getter(d,i):
    
    print(str(i))
    
    layer0_i = d.get(i).get('layer 0')
    
    return(layer0_i)



import multiprocessing as mp

k = d.keys() 

print("get layer 0")

pool = mp.Pool(mp.cpu_count())

layer0 = [pool.apply(layer0_getter, args=(d,i)) for i in k]

pool.close()

        
def layer1_getter(d,i):
    
    print(str(i))
  
    layer1_i = d.get(i).get('layer 1')
    
    return(layer1_i)

import multiprocessing as mp

k = d.keys() 

print("get layer 1")

pool = mp.Pool(mp.cpu_count())

layer1 = [pool.apply(layer1_getter, args=(d,i)) for i in k]

pool.close()

def layer2_getter(d,i):
    
    print(str(i))
  
    layer2_i = d.get(i).get('layer 2')
    
    return(layer2_i)

import multiprocessing as mp

k = d.keys() 

print("get layer 2")

pool = mp.Pool(mp.cpu_count())

layer2 = [pool.apply(layer2_getter, args=(d,i)) for i in k]

pool.close()

def layer3_getter(d,i):
    
    print(str(i))
  
    layer3_i = d.get(i).get('layer 3')
    
    return(layer3_i)

import multiprocessing as mp

k = d.keys() 

print("get layer 3")

pool = mp.Pool(mp.cpu_count())

layer3 = [pool.apply(layer3_getter, args=(d,i)) for i in k]

pool.close()

def layer4_getter(d,i):
    
    print(str(i))
  
    layer4_i = d.get(i).get('layer 4')
    
    return(layer4_i)

import multiprocessing as mp

k = d.keys() 

print("get layer 4")

pool = mp.Pool(mp.cpu_count())

layer4 = [pool.apply(layer4_getter, args=(d,i)) for i in k]

pool.close()

def layer5_getter(d,i):
    
    print(str(i))
  
    layer5_i = d.get(i).get('layer 5')
    
    return(layer5_i)

import multiprocessing as mp

k = d.keys() 

print("get layer 5")

pool = mp.Pool(mp.cpu_count())

layer5 = [pool.apply(layer5_getter, args=(d,i)) for i in k]

pool.close()

print("get x and y in parallel")

def x_y_getter(i):
    import numpy as np
    
    layer0 = i[0]
    layer1 = i[1]
    layer2 = i[2]
    layer3 = i[3]
    layer4 = i[4]
    layer5 = i[5]
    electron = i[6]
    
#    print("getting x and y values................................................")
    
#    for i in range(len(layer0)):
    if type(layer0)==type(None) or np.array(layer0).shape==(17,0):
        pass
    else:
        x0 = np.array(layer0)
        x0 = np.sum(x0,axis=0)
        y0 = np.array(electron)
#        beg=i
#        break
        
    if type(layer1)==type(None) or np.array(layer1).shape==(17,0):
        pass
    else:
        x1 = np.array(layer1)
        x1 = np.sum(x1,axis=0)
        y1 = np.array(electron)
        
    if type(layer2)==type(None) or np.array(layer2).shape==(17,0):
        pass
    else:
        x2 = np.array(layer0)
        x2 = np.sum(x2,axis=0)
        y2 = np.array(electron)
        
    if type(layer3)==type(None) or np.array(layer3).shape==(17,0):
        pass
    else:
        x3 = np.array(layer3)
        x3 = np.sum(x3,axis=0)
        y3 = np.array(electron)
        
    if type(layer4)==type(None) or np.array(layer4).shape==(17,0):
        pass
    else:
        x4 = np.array(layer4)
        x4 = np.sum(x4,axis=0)
        y4 = np.array(electron)
        
    if type(layer5)==type(None) or np.array(layer5).shape==(17,0):
        pass
    else:
        x5 = np.array(layer5)
        x5 = np.sum(x5,axis=0)
        y5 = np.array(electron)
        
        
    xlist = []
    ylist = []
    
    if 'x0' in locals():
        xlist.append(x0)
    if 'x1' in locals():
        xlist.append(x1)
    if 'x2' in locals():
        xlist.append(x2)
    if 'x3' in locals():
        xlist.append(x3)
    if 'x4' in locals():
        xlist.append(x4)
    if 'x5' in locals():
        xlist.append(x5)
        
    if 'y0' in locals():
        ylist.append(y0)
    if 'y1' in locals():
        ylist.append(y1)
    if 'y2' in locals():
        ylist.append(y2)
    if 'y3' in locals():
        ylist.append(y3)
    if 'y4' in locals():
        ylist.append(y4)
    if 'y5' in locals():
        xlist.append(y5)
            
    x = np.array(xlist)
    y = np.array(ylist)
    return((x,y))
    
# Parallelizing using Pool.apply()

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
xy = [pool.apply(x_y_getter, args=(i)) for i in (layer0,layer1,layer2,layer3,layer4,layer5,electron)]

# Step 3: Don't forget to close
pool.close() 

x = xy[0]
y = xy[1]

print("reshape x and y")

import numpy as np

x = np.reshape(x,(len(y),24))
x = x.astype('float32')

mu = np.mean(x)
x /= mu

print("pickling files")

import pickle
 
with open('/scratch/vljchr004/data/msc-thesis-data/x.pkl', 'wb') as x_file:
  pickle.dump(x, x_file)
  
with open('/scratch/vljchr004/msc-thesis-data/y.pkl', 'wb') as y_file:
  pickle.dump(y, y_file)


print("done.")






















