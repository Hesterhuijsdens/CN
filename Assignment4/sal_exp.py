import numpy as np
from equations_BM import *
from binary_data import *
import pickle
import pylab
import math


def poisson(lam, k):
    return np.divide(float(np.power(lam, k)), math.factorial(k)) * np.exp(-1. * lam)


for i in range(L):
    lst_new=np.array(lst2[i])
    min_E=0.5*(np.dot(lst_new.T, np.dot(W, lst_new))) + np.dot(theta.T,lst_new)
    E_list.append(min_E)
    countX=0
    for j in range(np.shape(s)[0]):
        if np.sum(s[j]-lst_new)==0:
            countX=countX+1
    observed_pattern.append(countX)
    iteration+=1
    print(iteration)


rate_obs=np.asarray(observed_pattern)/(np.shape(s)[0])
Z=np.sum(np.exp(E_list))
p_s=((1/Z)*np.exp(E_list))



