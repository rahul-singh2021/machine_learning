import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('insurance_data.csv')

x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1)

model=LogisticRegression()

model.fit(x_train,y_train)

model.predict(x_test)
