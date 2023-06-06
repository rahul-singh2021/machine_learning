import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")

model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)

from sklearn.externals import joblib

joblib.dump(model, 'model_joblib')

mj = joblib.load('model_joblib')
