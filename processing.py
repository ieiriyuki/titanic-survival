#!/usr/bin/python

from sklearn import preprocessing as prp
import pandas as pd
import random as rand

data = pd.read_csv("./data/train.csv", sep=",", header=0)

print("description: ", data.describe())
print("data shape is: {0}".format(data.shape))

print(data['Sex'].dtypes)

data = data.mask(data == 'female', 1)
data = data.mask(data == 'male', 0)
data['Sex'] = data['Sex'].astype('int64')

print(data['Sex'].dtypes)

print("{0}".format(data.loc[1]))
print("{0}".format(data.columns))
print(data.iloc[:,1:12].corr())

for col in data.columns:
    nacount = sum(pd.isna(data[col]))
    print("{0} has na: {1}".format(col, nacount))

train_data = data[['Survived','Pclass','Sex']]
print("shape changed: {0}".format(train_data.shape))

x = []
x.append(rand.sample(range(891), 33))

print(x[0])

#print(train_data['Sex'])

#end of file
