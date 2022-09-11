import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import pickle
import numpy as np

def get_X_from_file(file, full_adcs):
    with gzip.open(file,'rb') as f:
        data = pickle.load(f)
        x, y = data['first_bunch_x'], data['first_bunch_y']
        
        # in case that one of the 
        if x.shape[0] == 0 or y.shape[0] == 0 or x.shape != y.shape:
            return torch.nan * torch.zeros((0,1,2 * len(full_adcs)))
        
        x = x[full_adcs]
        y = y[full_adcs]

        X = np.stack( (x.to_numpy(),y.to_numpy()),0).astype(np.float16)
        # ({x,y},~600, BPMS)
        X = X - X.mean(1,keepdims = True)
        
        # removing bunches with nans
        bunches_without_nans = np.isfinite(X.sum((0,2)))
        X = X[:,bunches_without_nans,:]
        
        # other operations
        dim, N, vals = X.shape
        X = X.transpose((1,0,2))
        X = X.reshape((N,1,dim * vals))
        X = torch.tensor(X)
    return X