import numpy as np

def gradient_descent(x,y):
  w_curr=b_curr=0
  iterations=1000
  alpha=0.001
  m=len(x)

  for i in range(iterations):
    y_prd=w_curr*x+b_curr
    wd=(1/(2*m))*np.sum(x*(y_prd-y))
    bd=(1/(2*m))*np.sum(y_prd-y)
    w_curr=w_curr-alpha*wd
    b_curr=b_curr-alpha*bd
    print("w{},b{},iterations{}".format(w_curr,b_curr,i))

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y)
