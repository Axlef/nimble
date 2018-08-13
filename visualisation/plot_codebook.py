import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def load_codebook(hdf5_filename = 'codebook.hdf5'):
    with h5py.File(hdf5_filename, 'r') as hf5:
        codebook = hf5['codebook'][:]
        labels   = hf5['labels'][:]

        return (codebook, labels)

def load_data(hdf5_filename = 'data.hdf5'):
    with h5py.File(hdf5_filename, 'r') as hf5:
        descriptors = hf5['descriptors'][:]
        labels      = hf5['labels'][:]

        return (descriptors, labels)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=str, required = True)
    parser.add_argument("--data", type=str, required = True)
    args = parser.parse_args()

    (codebook, labels) = load_codebook(args.codebook)
    (X, y) = load_data(args.data)
    
    n_actions = 3

    main_title = {0:'Neutral', 1:'Clapping', 2:'Waving'}
    title_subplot = {0:'RelPos', 1:'JVel', 2:'JAcc', 3:'Ang', 4:'AngVel'}
    for action in range(1,n_actions):
        plt.figure()
        plt.suptitle(main_title[action])
        for i in range(3):
            plt.subplot(3, 1, i+1)        
            # for joint in range(15):
            #    plt.scatter(X[y == action, i*30+joint*2], X[y == action, i*30+joint*2+1], c = labels[y == action], s = 10, cmap = 'viridis')            
            
            plt.scatter(X[y == action, i*30+4*2], X[y == action, i*30+4*2+1], c = labels[y == action], s = 10, cmap = 'Set3')            
            #plt.scatter(X[y == action, i*30+10*2], X[y == action, i*30+10*2+1], c = labels[y == action], s = 10, cmap = 'Set3')            

            # plt.scatter(codebook[:,i*2], codebook[:,i*2+1], c = 'black', s = 30, alpha = 0.5)
            plt.title(title_subplot[i])
    
    plt.show()