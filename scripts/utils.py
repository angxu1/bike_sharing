import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def find_wasserstein(loc1,loc2,w1,w2):
    '''
    Compute the second moment Wasserstein distance between two sets of weighted locations.
    loc1,loc2: location sets that contains the coordinate of each location 
    w1,w2: weight vectors of the corresponding location set
    '''
    w1_len = w1.shape[0]
    w2_len = w2.shape[0]
    loc1 = loc1.reshape(1,-1,2)
    loc2 = loc2.reshape(-1,1,2)
    dist_mat = (loc1[:,:,0]-loc2[:,:,0])**2+(loc1[:,:,1]-loc2[:,:,1])**2
    c = dist_mat.reshape(-1)
    A = np.zeros((w1_len+w2_len,w1_len*w2_len))
    b = np.concatenate((w1,w2))
    for i in range(w1_len):
        A[i,w2_len*i:w2_len*(i+1)] = np.ones(w2_len)
    for i in range(w2_len):
        A[i+w1_len,np.linspace(i,(w1_len-1)*w2_len+i,w1_len,dtype=int)] = np.ones(w1_len)
    lam_bounds = (-2e-7,None)
    res = linprog(c, A_eq=A, b_eq=b, bounds=[lam_bounds]*(w1_len*w2_len))
    return np.sqrt(res.fun),res.x

