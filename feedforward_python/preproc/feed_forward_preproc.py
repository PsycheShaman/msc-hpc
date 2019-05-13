 
print("starting")

import glob

print("imported glob")

files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/unprocessed" + '/**/*.txt', recursive=True)

a = list(range(1,len(files)-1))

files_in_order = []
for i in a:
    files_in_order.append(files[i])

print("read files list")

#def parallel_proc(i):
        
from ast import literal_eval

d = {}

#for i in range(0,len(files_in_order)):
#practice run on 5 files:
for i in range(0,5):
    print(files_in_order[i])
    di = open(files_in_order[i])
#di = open(i)
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
            
def layer_and_pdg(d,i):
    
    print(str(i))
    
#    pdgCode = []
#    
#    layer0 = []
#    
#    layer1 = []
#    
#    layer2 = []
#    
#    layer3 = []
#    
#    layer4 = []
#    
#    layer5 = []
    
#    print("extracting data from dictionaries................................................")
    

    pdgCode_i = d.get(i).get('pdgCode')    
    layer0_i = d.get(i).get('layer 0')
    layer1_i = d.get(i).get('layer 1')
    layer2_i = d.get(i).get('layer 2')
    layer3_i = d.get(i).get('layer 3')
    layer4_i = d.get(i).get('layer 4')
    layer5_i = d.get(i).get('layer 5')

#    pdgCode.append(pdgCode_i)
#    
#    layer0.append(layer0_i)
#    layer1.append(layer1_i)
#    layer2.append(layer2_i)
#    layer3.append(layer3_i)
#    layer4.append(layer4_i)
#    layer5.append(layer5_i)
    
    return((pdgCode_i,layer0_i,layer1_i,layer2_i,layer3_i,layer4_i,layer5_i))

            
# Parallelizing using Pool.apply()

print("layer and pdg")

import multiprocessing as mp

## Step 1: Init multiprocessing.Pool()
#pool = mp.Pool(mp.cpu_count())
#
## Step 2: `pool.apply` the `howmany_within_range()`
#dict1 = [pool.apply(parallel_proc, args=(i)) for i in files_in_order]
#
## Step 3: Don't forget to close
#pool.close()  

k = d.keys() 

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
lay_pdg = [pool.apply(layer_and_pdg, args=(d,i)) for i in k]

# Step 3: Don't forget to close
pool.close()

print("Get elements from layer and pdg function")

pdgCode = lay_pdg[0]

layer0 = lay_pdg[1]

layer1 = lay_pdg[2]

layer2 = lay_pdg[3]

layer3 = lay_pdg[4]

layer4 = lay_pdg[5]

layer5 = lay_pdg[6] 

electron = []

print("electron one-hot encoding")

for i in pdgCode:
    if abs(i)==11:
        electron.append(1)
    else:
        electron.append(0)

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
        
#    for i in range(beg+1,len(layer0)):
#        if type(layer0[i])==type(None) or np.array(layer0[i]).shape==(17,0):
#            continue
#        else:
#            xi = np.array(layer0[i])
#            xi = np.sum(xi,axis=0)
#            yi = electron[i]
#            x = np.concatenate((x,xi))
#            y = np.append(y,yi)
#        if type(layer1[i])==type(None) or np.array(layer1[i]).shape==(17,0):
#            continue
#        else:
#            xi = np.array(layer1[i])
#            xi = np.sum(xi,axis=0)
#            yi = electron[i]
#            x = np.concatenate((x,xi))
#            y = np.append(y,yi)
#        if type(layer2[i])==type(None) or np.array(layer2[i]).shape==(17,0):
#            continue
#        else:
#            xi = np.array(layer2[i])
#            xi = np.sum(xi,axis=0)
#            yi = electron[i]
#            x = np.concatenate((x,xi))
#            y = np.append(y,yi)
#        if type(layer3[i])==type(None) or np.array(layer3[i]).shape==(17,0):
#            continue
#        else:
#            xi = np.array(layer3[i])
#            xi = np.sum(xi,axis=0)
#            yi = electron[i]
#            x = np.concatenate((x,xi))
#            y = np.append(y,yi)
#        if type(layer4[i])==type(None) or np.array(layer4[i]).shape==(17,0):
#            continue
#        else:
#            xi = np.array(layer4[i])
#            xi = np.sum(xi,axis=0)
#            yi = electron[i]
#            x = np.concatenate((x,xi))
#            y = np.append(y,yi)
#        if type(layer5[i])==type(None) or np.array(layer5[i]).shape==(17,0):
#            continue
#        else:
#            xi = np.array(layer5[i])
#            xi = np.sum(xi,axis=0)
#            yi = electron[i]
#            x = np.concatenate((x,xi))
#            y = np.append(y,yi)
#        print(str(100*i/len(layer0)))
        
#        [x0,x1,x2,x3,x4,x5]
#        [y0,y1,y2,y3,y4,y5]
        
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






















