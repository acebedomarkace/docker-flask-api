# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:44:58 2019

@author: 10011881
"""

#  Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


#  loading the dataset
iris = load_iris()
X = iris.data
y = iris.target


#  split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.5)


#  build the model
clf = RandomForestClassifier(n_estimators=10)


#  train the classifier
clf.fit(X_train, y_train)


#  Predictions
predicted = clf.predict(X_test)


#  Check accuracy
print(accuracy_score(predicted, y_test))


# Pickling the model
import pickle
with open('/Users/10011881/SAAS/rf.pkl','wb') as model_pkl:
    pickle.dump(clf, model_pkl)

