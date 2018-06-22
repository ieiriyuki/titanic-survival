#!/usr/bin/python

from sklearn import preprocessing as prp
import pandas as pd
import numpy as np

a = 1.
b = 3.
c = a * b
print("{0}".format(c))

data = pd.read_csv("./data/train.csv", sep=",", header=0)

print(data.describe())
print("data shape is: {0}".format(data.shape))

print("{0}".format(data.loc[1]))
print("{0}".format(data.columns))
print(data.loc[:,['Pclass','Age','Fare']].corr())

'''
for col in data.columns:
    nacount = sum(pd.isna(data[col]))
    print("{0} has na: {1}".format(col, nacount))
'''

train_data = data[['Survived','Pclass','Sex']]
print("shape changed: {0}".format(train_data.shape))


sex = train_data['Sex']
print(sex == 'female')

sex = sex[]

#end of file
