import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('carprices.csv')

x = df[['Mileage','Age(yrs)']]
y = df['Sell Price($)']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15)

clf=LinearRegression()

clf.fit(x_train,y_train)

clf.predict(x_test)
