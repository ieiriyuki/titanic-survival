#!/usr/bin/python

from sklearn.linear_morel import Logistic_Regression as logis
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prp
import pandas as pd

data0 = pd.read_csv("./data/train.csv", sep=",", header=0)

data1 = data0[['Survived','Pclass','Sex']]

mdl = Logistic_Regression()

x_train, y_train, x_test, y_test = train_test_split(
    data0[['Pclass','Sex']], data0[['Survived']], test_size=0.2
)

mdl.fit(x_train, y_train)

print(mdl.coef_)
print(mdl.intercept_)

#end of file
