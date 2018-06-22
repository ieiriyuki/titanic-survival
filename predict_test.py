#!/usr/bin/python

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from time import time
import pandas as pd

train0 = pd.read_csv("./data/train.csv", sep=",", header=0)
test0 = pd.read_csv("./data/test.csv", sep=",", header=0)

train1 = train0[['Survived','Pclass','Sex', 'Parch']]
train1 = train1.mask(train1 == 'female', 1)
train1 = train1.mask(train1 == 'male', 0)

test1 = test0[['Pclass','Sex', 'Parch']]
test1 = test1.mask(test1 == 'female', 1)
test1 = test1.mask(test1 == 'male', 0)

mdl = LogisticRegression(C=0.2)

x_train, x_test, y_train, y_test = train_test_split(
    train1[['Pclass','Sex','Parch']], train1[['Survived']], test_size=0.2
)

mdl.fit(x_train, y_train)

print("score for train is: {0}".format(mdl.score(x_train, y_train)))
print("score for test is: {0}".format(mdl.score(x_test, y_test)))

pred = mdl.predict(test1[['Pclass','Sex', 'Parch']])

submit = pd.DataFrame({'PassengerId': test0['PassengerId'],
                        'Survived': pred})

submit.to_csv("./data/pc_sx_pr_submission.csv", index=False)

#end of file
