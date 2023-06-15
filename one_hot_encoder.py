import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df=pd.read_csv('/content/homeprices (1).csv')

le=LabelEncoder()

dfle=df

dfle.town=le.fit_transform(dfle.town)

x=dfle[['town','area']].values
y=dfle.price

ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')

x=ct.fit_transform(x)
x=x[:,1:]

model.fit(x,y)
