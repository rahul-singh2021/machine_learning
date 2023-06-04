import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")

model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)

import pickle

with open('model_pickle','wb') as file:
    pickle.dump(model,file)
    
with open('model_pickle','rb') as file:
    mp = pickle.load(file)    
