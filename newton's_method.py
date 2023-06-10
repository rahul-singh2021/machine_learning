%matplotlib inline

import numpy as np
import pandas as pd
from patsy import dmatrices
import warnings

def sigmoid(x):
    '''Sigmoid function of x.'''
    return 1/(1+np.exp(-x))

np.random.seed(0)
tol=1e-8
lam = None
max_iter = 20 

r = 0.95
n = 1000  
sigma = 1

beta_x, beta_z, beta_v = -4, .9, 1
var_x, var_z, var_v = 1, 1, 4

formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'

x, z = np.random.multivariate_normal([0,0], [[var_x,r],[r,var_z]], n).T

v = np.random.normal(0,var_v,n)**3

A = pd.DataFrame({'x' : x, 'z' : z, 'v' : v})
A['log_odds'] = sigmoid(A[['x','z','v']].dot([beta_x,beta_z,beta_v]) + sigma*np.random.normal(0,1,n))



A['y'] = [np.random.binomial(1,p) for p in A.log_odds]


y, X = dmatrices(formula, A, return_type='dataframe')

X.head(100)
