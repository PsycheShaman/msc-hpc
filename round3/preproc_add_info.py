lsprint("==============================================================================================")

#import argparse
#
#parser = argparse.ArgumentParser()
#parser.add_argument("run", help="enter the specific run you need to process",type=str)
#args = parser.parse_args()
#
#run = str(args.run)

print("starting........................................................................................")

import glob

print("imported glob........................................................................................")

run = '000265309'

files_in_order = glob.glob("/scratch/vljchr004/data/msc-thesis-data/unprocessed/" + run + '/**/*.txt', recursive=True)

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
        P = [di.get(k).get('P') for k in ki]
        l = [di.get(k).get('layer0') for k in ki]
        
        return(P)
        


import numpy as np

print("pdg........................................................................................")
        
P0 = [file_reader1(i) for i in files_in_order]



print("layer 0........................................................................................")

layer0 = [file_reader2(i,"layer 0") for i in files_in_order]

layer0 = np.array([item for sublist in layer0 for item in sublist])

P0 = np.array([item for sublist in P0 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer0])

layer0 = np.delete(layer0, empties)

layer0 = np.stack(layer0)

P0 = np.delete(P0, empties)

nz = np.array([np.count_nonzero(i) for i in layer0])

zeros = np.where(nz==0)

del(layer0)

P0 = np.delete(P0,zeros)

print("layer 1........................................................................................")

layer1 = [file_reader2(1,"layer 1") for i in files_in_order]

layer1 = np.array([item for sublist in layer1 for item in sublist])

P1 = np.array([item for sublist in P1 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer1])

layer1 = np.delete(layer1, empties)

layer1 = np.stack(layer1)

P1 = np.delete(P1, empties)

nz = np.array([np.count_nonzero(i) for i in layer1])

zeros = np.where(nz==0)

del(layer1)

P1 = np.delete(P1,zeros)

print("layer 0........................................................................................")

layer2 = [file_reader2(i,"layer 2") for i in files_in_order]

layer2 = np.array([item for sublist in layer2 for item in sublist])

P2 = np.array([item for sublist in P2 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer2])

layer2 = np.delete(layer2, empties)

layer2 = np.stack(layer2)

P2 = np.delete(P2, empties)

nz = np.array([np.count_nonzero(i) for i in layer2])

zeros = np.where(nz==0)

del(layer2)

P2 = np.delete(P2,zeros)

print("layer 0........................................................................................")

layer3 = [file_reader2(i,"layer 3") for i in files_in_order]

layer3 = np.array([item for sublist in layer3 for item in sublist])

P3 = np.array([item for sublist in P3 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer3])

layer3 = np.delete(layer3, empties)

layer3 = np.stack(layer3)

P3 = np.delete(P3, empties)

nz = np.array([np.count_nonzero(i) for i in layer3])

zeros = np.where(nz==0)

del(layer3)

P3 = np.delete(P3,zeros)

print("layer 0........................................................................................")

layer0 = [file_reader2(i,"layer 0") for i in files_in_order]

layer0 = np.array([item for sublist in layer0 for item in sublist])

P0 = np.array([item for sublist in P0 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer0])

layer0 = np.delete(layer0, empties)

layer0 = np.stack(layer0)

P0 = np.delete(P0, empties)

nz = np.array([np.count_nonzero(i) for i in layer0])

zeros = np.where(nz==0)

del(layer0)

P0 = np.delete(P0,zeros)

##########################################

pdgCode = np.concatenate([pdgCode0,pdgCode1,pdgCode2,pdgCode3,pdgCode4,pdgCode5]).ravel()
  
np.save('/scratch/vljchr004/data/msc-thesis-data/cnn/y_' + run + '.npy',y,allow_pickle=False)
np.save('/scratch/vljchr004/data/msc-thesis-data/cnn/x_' + run + '.npy',x,allow_pickle=False)

print("done.........................................................................................")

print("==============================================================================================")
