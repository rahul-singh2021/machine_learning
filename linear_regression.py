import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("homeprices.csv")

%matplotlib inline
plt.xlabel("area(in square feet)")
plt.ylabel("prices(in $)")
plt.scatter(df.area,df.price,color='r',marker='*')

reg=linear_model.LinearRegression()
reg.fit(df[["area"]],df.price)

%matplotlib inline
plt.xlabel("area(in square feet)")
plt.ylabel("prices(in $)")
plt.scatter(df.area,df.price,color='r',marker='*')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
