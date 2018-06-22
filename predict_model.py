#!/usr/bin/python

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd

data0 = pd.read_csv("./data/train.csv", sep=",", header=0)

data1 = data0[['Survived','Pclass','Sex', 'Parch']]
data1 = data1.mask(data1 == 'female', 1)
data1 = data1.mask(data1 == 'male', 0)

mdl = LogisticRegression()

params_dist

x_train, x_test, y_train, y_test = train_test_split(
    data1[['Pclass','Sex','Parch']], data1[['Survived']], test_size=0.2
)

print("a",x_train.shape)
print("b",y_train.shape)

mdl.fit(x_train, y_train)

print(mdl.coef_)
print(mdl.intercept_)

print("score for train is: {0}".format(mdl.score(x_train, y_train)))
print("score for test is: {0}".format(mdl.score(x_test, y_test)))

'''
for i in range(5):

'''

#end of file
