import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv('/content/homeprices (1).csv')

dummies=pd.get_dummies(df.town)

merge=pd.concat([df,dummies],axis='columns')
final=merge.drop(['town','west windsor'],axis='columns')

model=LinearRegression()

x=final.drop('price',axis='columns')

y=final.price

model.fit(x,y)
