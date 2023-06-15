%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits=load_digits()

plt.gray()

digits.target[0:5]

x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)

model=LogisticRegression()
model.fit(x_train,y_train)

model.predict(digits.data[0:5])
