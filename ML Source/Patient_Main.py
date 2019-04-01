from ClassifyNB import classifyNB
from ClassifySVM import classifySVM
from ClassifyRFC import classifyRFC
from ClassifyLREG import classifyLREG
from ClassifyK_NN import classifyK_NN
from sklearn.metrics import accuracy_score
from EvaluationMetrics import Accuracy

import numpy as np
import pylab as pl
import pandas as pd
import scikitplot as skplt

# 6000 data
train_data = pd.read_excel("PD Wrangled Data/Combined Data.xlsx", "Filtered")
test_data = pd.read_excel("PD Wrangled Data/Combined Data.xlsx", "Filtered")

train_row_st = 1
train_row_end = 241
train_col_st = 1
train_col_end = 16

test_row_st = 242
test_row_end = 481
test_col_st = 1
test_col_end = 16

label_col = 17

df1 = pd.DataFrame(test_data)
testData_features = df1[test_row_st:test_row_end][df1.columns[test_col_st:train_col_end]].values.tolist()
testData_label = df1[test_row_st:test_row_end][df1.columns[label_col]].values.tolist()

df2 = pd.DataFrame(train_data)
trainData_features = df2[train_row_st:train_row_end][df2.columns[train_col_st:train_col_end]].values.tolist()
trainData_label =  df2[train_row_st:train_row_end][df2.columns[label_col]].values.tolist()

# Naive Bayse
clfNB = classifyNB(trainData_features, trainData_label)
predNB = clfNB.predict(testData_features)
accuracyNB = accuracy_score(predNB, testData_label)
print("accuracyNB",accuracyNB, sep=" : ")

#SVM
clfSVM = classifySVM(trainData_features, trainData_label)
predSVM = clfSVM.predict(testData_features)
accuracySVM = accuracy_score(predSVM, testData_label)
print("accuracySVM",accuracySVM, sep=" : ")

#Random Forest Classifier
clfRFC = classifyRFC(trainData_features, trainData_label)
predRFC = clfRFC.predict(testData_features)
accuracyRFC = accuracy_score(predRFC, testData_label)
print("accuracyRFC",accuracyRFC, sep=" : ")

# Linear Regression
clfLREG = classifyLREG(trainData_features, trainData_label)
predLREG = clfLREG.predict(testData_features)
accuracyLREG = accuracy_score(predLREG, testData_label)
print("accuracyLREG",accuracyLREG, sep=" : ")

# K-NN
clfK_NN = classifyK_NN(trainData_features, trainData_label)
predK_NN = clfK_NN.predict(testData_features)
accuracyK_NN = accuracy_score(predK_NN, testData_label)
print("accuracyK_NN",accuracyK_NN, sep=" : ")