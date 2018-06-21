#!/usr/bin/python

import pandas as pd

data = pd.read_csv("./data/train.csv", sep=",", header=0)

a = 1.
b = 3.
c = a * b

print(data.describe())

print("{0}".format(c))

print("{0}".format(data.loc[1]))
