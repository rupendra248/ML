# -*- coding: utf-8 -*-
"""5_ML_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AVJRTCE83aB0HmAFQMzFHadX1eNauxX-

# **ML Lab Assignment:5**

#**Solution:2**
"""

# Commented out IPython magic to ensure Python compatibility.
#import required packages
# %matplotlib inline
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns
from sklearn import preprocessing

# Inputs and labels
X = [[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,1],[-2,-2]]
Y = [1,1,2,2,3,3,4,4]

"""# **On the Train Data:**"""

# FUNCTION TO CALCULATE OUTPUT BEFORE APPLYING ACTIVATION FUNCTION 
def calculate_yin(x1, x2, w1, w2, w3, w4, b1, b2):
  yin1 = w1 * x1 + w2 * x2  + b1 # sum of weighted multiplication of inputs and weights
  yin2 = w3 * x1 + w4 * x2  + b2 # sum of weighted multiplication of inputs and weights
  # print(x1, x2, w1, w2, w3, w4, b1, b2)
  return yin1, yin2

# FUNCTION TO UPDATE WEIGHTS
def update_weights(w1, x1, w2, x2, w3, w4, lr, z):
  w1_new = w1 + lr * z[0] * x1   # updating weight1
  w2_new = w2 + lr * z[0] * x2   # updating weight2
  w3_new = w3 + lr * z[1] * x1   # updating weight1
  w4_new = w4 + lr * z[1] * x2   # updating weight2
 
  return w1_new, w2_new, w3_new, w4_new
  
def class_to_category(y):
  if y == 1: return np.array([0, 0])
  if y == 2: return np.array([0, 1])
  if y == 3: return np.array([1, 0])
  return np.array([1, 1])

# ACTIVATION FUNCTION
def activation_fn(yin):
  sigmoid = (1/(1 + np.exp(-yin)))
  return sigmoid

def train_function(w1, w2, w3, w4, b1, b2, lr):
  # SET ALL PARAMETERS AND VARIABLES
  x = X # input 
  target = Y # output

  threshold1 = 0.5
  threshold2 = 0.5 # threshold
  epoch = 20 # epoch

  # BASIC PERCEPTRON LEARNING PROCESS
  for e in range(epoch): # loop for each epoch
    out = []
    for i in range(len(x)): # loop for each row in x
      x1 = float(x[i][0])
      x2 = float(x[i][1])

      yin1, yin2 = calculate_yin(x1, x2, w1, w2, w3, w4, b1, b2) # calculate yin
      yact1 = activation_fn(yin1) 
      yact2 = activation_fn(yin2) 

      if yact1 > threshold1 and yact2 > threshold2: 
        y = 4
      elif  yact1 > threshold1 and yact2 < threshold2: 
        y = 3
      elif yact1 < threshold1 and yact2 > threshold2: 
        y = 2
      else: 
        y = 1

      # check if predicted output and target is different then update weight
      if y != target[i]: 
        loss = class_to_category(target[i]) - class_to_category(y)
        w1, w2, w3, w4 = update_weights(w1, x1, w2, x2, w3, w4, lr, loss)
      out.append(y) # append all prediction in 'out'

  # performance evaluation
  accuracy = accuracy_score(target,out)
  print("Weights:[{:.2f},{:.2f},{:.2f},{:.2f}] and Accuracy:{:.2f}".format(w1,w2,w3,w4,accuracy))
  return accuracy

"""# Parameters Tuning:"""

w1_a = [-3, -2, 0, 1]
w2_a = [-2, -1, 1, 2]
w3_a = [-2, -1, 1, 2]
w4_a = [-2, -1, 0, 2]
b1_a = [-1, 1]
b2_a = [-1, 1]
lr_a = [0.001, 0.01, 1]

max_accuracy = 0
w1_max = 0
w2_max = 0
w3_max = 0
w4_max = 0
b1_max = 0
b2_max = 0
lr_max = 0

for w1 in w1_a:  
  for w2 in w2_a: 
    for w3 in w3_a:
      for w4 in w4_a:
        for b1 in b1_a:
          for b2 in b2_a: 
            for lr in lr_a: 
              accuracy = train_function(w1, w2, w3, w4, b1, b2, lr)
              if accuracy > max_accuracy:
                max_accuracy = accuracy
                w1_max = w1
                w2_max = w2
                w3_max = w3
                w4_max = w4
                b1_max = b1
                b2_max = b2
                lr_max = lr

"""# best parameters:"""

max_accuracy, w1_max, w2_max, w3_max, w4_max, b1_max, b2_max, lr_max

"""# **On the Test Data:**"""

# test dataset
X_test = [[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,1],[-2,-2], [-3,-2],[-4,-2]]
y_test = [1, 1, 1, 1, 1, 4, 1, 3, 3, 4]

def test_function(w1, w2, w3, w4, b1, b2):
  # SET ALL PARAMETERS AND VARIABLES
  x = X_test # input 
  target = y_test # output

  # threshold
  threshold1 = 0.5
  threshold2 = 0.5 

  # BASIC PERCEPTRON LEARNING PROCESS
  out_test = []
  for i in range(len(x)): # loop for each row in x
    x1 = float(x[i][0])
    x2 = float(x[i][1])

    yin1, yin2 = calculate_yin(x1, x2, w1, w2, w3, w4, b1, b2) # calculate yin
    yact1 = activation_fn(yin1) 
    yact2 = activation_fn(yin2) 

    if yact1 > threshold1 and yact2 > threshold2: 
      y = 4
    elif  yact1 > threshold1 and yact2 < threshold2: 
      y = 3
    elif yact1 < threshold1 and yact2 > threshold2: 
      y = 2
    else: 
      y = 1
    out_test.append(y) # append all prediction in 'out'

  print("\n TEST RESULTS:\n")
  print("Final weights:{},{},{},{}".format(w1,w2,w3,w4))
  print("Final predicted output:",out_test)

  # performance evaluation
  accuracy = accuracy_score(y_test,out_test)
  print("accuracy:",accuracy)

  return accuracy

test_function(w1_max, w2_max, w3_max, w4_max, b1_max, b2_max)