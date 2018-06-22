#!/usr/bin/python

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from time import time
import pandas as pd

data0 = pd.read_csv("./data/train.csv", sep=",", header=0)

data1 = data0[['Survived','Pclass','Sex', 'Parch']]
data1 = data1.mask(data1 == 'female', 1)
data1 = data1.mask(data1 == 'male', 0)

mdl = LogisticRegression()

kf = KFold(n_splits=9)

param_grid = {'C': [0.2,0.4,0.6,0.8,1.0]}

grid_search = GridSearchCV(mdl, param_grid=param_grid, cv=kf, return_train_score=True)

x_train, x_test, y_train, y_test = train_test_split(
    data1[['Pclass','Sex','Parch']], data1[['Survived']], test_size=0.2
)

start = time()
grid_search.fit(x_train, y_train)
print("random search tool time : {0}".format(time() - start))

print("mean train score", grid_search.cv_results_['mean_train_score'])
print("best params", grid_search.best_params_['C'])

#print("a",x_train.shape)
#print("b",y_train.shape)
mdl2 = LogisticRegression(C=0.2)

mdl2.fit(x_train, y_train)


print(mdl2.coef_)
print(mdl2.intercept_)

print("score for train is: {0}".format(mdl2.score(x_train, y_train)))
print("score for test is: {0}".format(mdl2.score(x_test, y_test)))


'''
for i in range(5):

'''

#end of file
