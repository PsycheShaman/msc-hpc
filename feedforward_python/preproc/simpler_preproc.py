import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run", help="enter the specific run you need to process",type=str)
args = parser.parse_args()

run = str(args.run)

print("starting........................................................................................")

import glob

print("imported glob........................................................................................")

files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/unprocessed/" + run + '/**/*.txt', recursive=True)

a = list(range(1,len(files)-1))

files_in_order = [files[i] for i in a]

print("read files list........................................................................................")

from ast import literal_eval

def file_reader(i):
    di = open(i)
    di = di.read()
    if di == "}":
        pass
    else:
        di = di + "}"
        di = literal_eval(di)
        ki = list(di.keys())
        pdgCode = [di.get(k).get('pdgCode') for k in ki]
        layer0 = [di.get(k).get('layer0') for k in ki]
        layer1 = [di.get(k).get('layer1') for k in ki]
        layer2 = [di.get(k).get('layer2') for k in ki]
        layer3 = [di.get(k).get('layer3') for k in ki]
        layer4 = [di.get(k).get('layer4') for k in ki]
        layer5 = [di.get(k).get('layer5') for k in ki]
        return((pdgCode,layer0,layer1,layer2,layer3,layer4,layer5))

import multiprocessing as mp

pool = mp.Pool(mp.cpu_count())

d = pool.map(file_reader, files_in_order)

pool.close()

print("mapped out files to useful elements....................................................................")
print("this is pdg............................................................................................")

pdgCode = d[0]

pdgCode = pdgCode[0]

print(pdgCode)

layer0 = d[1]

print("this is layer 0 before transform............................................................................................")

print(layer0)

layer0 = layer0[0]

print("this is layer 0 before transform............................................................................................")

print(layer0)

layer1 = d[2]
layer1 = layer1[0]

layer2 = d[3]
layer2 = layer2[0]

layer3 = d[4]
layer3 = layer3[0]

layer4 = d[5]
layer4 = layer4[0]

layer5 = d[6]
layer5 = layer5[0]

import numpy as np

def pdg_code_to_elec(i):
    if np.abs(i)==11:
        return(1)
    else:
        return(0)
        
#electron = [pdg_code_to_elec(i) for i in pdgCode]

pool = mp.Pool(mp.cpu_count())

electron = pool.map(pdg_code_to_elec,pdgCode)

pool.close()

print("mapped out electrons....................................................................")

#TODO: look at the rest

def x_0_getter(i):
    import numpy as np

    layer0 = i
    if type(layer0)==type(None) or np.array(layer0).shape==(17,0):
        pass
    else:
        x0 = np.array(layer0)
        x0 = np.sum(x0,axis=0)

    if 'x0' in locals():
        return(x0)
#

print("get y from layer 0........................................................................................")

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

pool = mp.Pool(mp.cpu_count())

x0 = pool.map(x_0_getter, layer0)

pool.close()

pool = mp.Pool(mp.cpu_count())

y0 = pool.map(y_0_getter, layer0)

pool.close()

print("layer 1........................................................................................")

pool = mp.Pool(mp.cpu_count())

x1 = pool.map(x_0_getter, layer1)

pool.close()

pool = mp.Pool(mp.cpu_count())

y1 = pool.map(y_0_getter, layer1)

pool.close()

print("layer 2........................................................................................")

pool = mp.Pool(mp.cpu_count())

x2 = pool.map(x_0_getter, layer2)

pool.close()

pool = mp.Pool(mp.cpu_count())

y2 = pool.map(y_0_getter, layer2)

pool.close()

print("layer 3........................................................................................")

pool = mp.Pool(mp.cpu_count())

x3 = pool.map(x_0_getter, layer3)

pool.close()

pool = mp.Pool(mp.cpu_count())

y3 = pool.map(y_0_getter, layer3)

pool.close()

print("layer 4........................................................................................")

pool = mp.Pool(mp.cpu_count())

x4 = pool.map(x_0_getter, layer4)

pool.close()

pool = mp.Pool(mp.cpu_count())

y4 = pool.map(y_0_getter, layer4)

pool.close()

print("layer 5........................................................................................")

pool = mp.Pool(mp.cpu_count())

x5 = pool.map(x_0_getter, layer5)

pool.close()

pool = mp.Pool(mp.cpu_count())

y5 = pool.map(y_0_getter, layer5)

pool.close()


#pool = mp.Pool(mp.cpu_count())
#
#y0 = [pool.apply(y_0_getter, args=(i,electron)) for i in (layer0)]
#
#pool.close()
#
#pool = mp.Pool(mp.cpu_count())
#
#print("layer 1........................................................................................")
#
#x1 = [pool.apply(x_0_getter, args=(i)) for i in (layer1)]
#
#pool.close()
#
#
#pool = mp.Pool(mp.cpu_count())
#
#y1 = [pool.apply(y_0_getter, args=(i,electron)) for i in (layer1)]
#
#pool.close()
#
#print("layer 2........................................................................................")
#
#pool = mp.Pool(mp.cpu_count())
#
#x2 = [pool.apply(x_0_getter, args=(i)) for i in (layer2)]
#
#pool.close()
#
#
#pool = mp.Pool(mp.cpu_count())
#
#y2 = [pool.apply(y_0_getter, args=(electron,i)) for i in (layer2)]
#
#pool.close()
#
#print("layer 3........................................................................................")
#
#pool = mp.Pool(mp.cpu_count())
#
#x3 = [pool.apply(x_0_getter, args=(i)) for i in (layer3)]
#
#pool.close()
#
#
#pool = mp.Pool(mp.cpu_count())
#
#y3 = [pool.apply(y_0_getter, args=(electron,i)) for i in (layer3)]
#
#pool.close()
#
#print("layer 4........................................................................................")
#
#pool = mp.Pool(mp.cpu_count())
#
#x4 = [pool.apply(x_0_getter, args=(i)) for i in (layer4)]
#
#pool.close()
#
#
#pool = mp.Pool(mp.cpu_count())
#
#y4 = [pool.apply(y_0_getter, args=(electron,i)) for i in (layer4)]
#
#pool.close()
#
#print("layer 5........................................................................................")
#
#pool = mp.Pool(mp.cpu_count())
#
#x5 = [pool.apply(x_0_getter, args=(i)) for i in (layer5)]
#
#pool.close()
#
#
#pool = mp.Pool(mp.cpu_count())
#
#y5 = [pool.apply(y_0_getter, args=(electron,i)) for i in (layer5)]
#
#pool.close()

#

print("concatenating........................................................................................")

x = np.concatenate((x0,x1,x2,x3,x4,x5),axis=None)

y = np.concatenate((y0,y1,y2,y3,y4,y5),axis=None)

print("reshape x and y........................................................................................")

import numpy as np

x = np.reshape(x,(len(y),24))
x = x.astype('float32')

mu = np.mean(x)
x /= mu

print("pickling files........................................................................................")

import pickle

with open('/scratch/vljchr004/data/msc-thesis-data/x_' + run + '.pkl', 'wb') as x_file:
  pickle.dump(x, x_file)

with open('/scratch/vljchr004/msc-thesis-data/y_' + run + '.pkl', 'wb') as y_file:
  pickle.dump(y, y_file)


print("done.........................................................................................")
