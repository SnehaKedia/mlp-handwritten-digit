#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Imports

import sys
import numpy as np
import pandas as pd
import time
from numpy import genfromtxt


# In[2]:


## Record Time
startTime = time.time()


# In[3]:


## Sigmoid Activation Function

def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s

## Cross Entropy Loss Funtion
def crossEntropyLoss(X, Y):

    summ = np.sum(np.multiply(X, np.log(Y)))
    m = X.shape[1]
    cost = -(1/m) * summ

    return cost

## Softmax Activation Function
def softmax(x):
    s = np.exp(x) / np.sum(np.exp(x), axis = 0)
    return s

## Forward Feed Pass
def feedForwardPass(X, parameters):

    output = {}
    output['Z1'] = np.matmul(parameters['W1'], X) + parameters['b1']
    output['A1'] = sigmoid(output['Z1'])
    output['Z2'] = np.matmul(parameters['W2'], output['A1']) + parameters['b2']
    output['A2'] = sigmoid(output['Z2'])
    output['Z3'] = np.matmul(parameters['W3'], output['A2']) + parameters['b3']
    output['A3'] = softmax(output['Z3'])

    return output

## Backward Propagation
def backPropagation(X, Y, parameters, output):
    
    ## Calculating differentiation of all parameters, updating the gradients and storing in the dictionary

    gradZ3 = output['A3'] - Y
    gradW3 = (1./mBatch) * np.matmul(gradZ3, output['A2'].T)
    gradb3 = (1./mBatch) * np.sum(gradZ3, axis = 1, keepdims = True)
    
    gradA2 = np.matmul(parameters['W3'].T, gradZ3)
    gradZ2 = gradA2 * sigmoid(output['Z2']) * (1 - sigmoid(output['Z2']))
    gradW2 = (1./mBatch) * np.matmul(gradZ2, output["A1"].T)
    gradb2 = (1./mBatch) * np.sum(gradZ2, axis=1, keepdims = True)

    gradA1 = np.matmul(parameters['W2'].T, gradZ2)
    gradZ1 = gradA1 * sigmoid(output['Z1']) * (1 - sigmoid(output['Z1']))
    gradW1 = (1./mBatch) * np.matmul(gradZ1, X.T)
    gradb1 = (1./mBatch) * np.sum(gradZ1, axis = 1, keepdims = True)

    gradient = {'gradW1': gradW1, 'gradW2': gradW2, 'gradW3': gradW3, 'gradb1': gradb1, 'gradb2': gradb2, 'gradb3': gradb3}

    return gradient


# In[4]:


# ## Read inputs from the file (in Dataframe using pandas)

# ## X -> Images, Y -> Labels

# n = len(sys.argv)

# if n == 4:
#     Xtrain = pd.read_csv(sys.argv[1], header=None)
#     Ytrain = pd.read_csv(sys.argv[2], header=None)
#     Xtest = pd.read_csv(sys.argv[3], header=None)
# else:
#     Xtrain = pd.read_csv('train_image.csv', header=None)
#     Ytrain = pd.read_csv('train_label.csv', header=None)
#     Xtest = pd.read_csv('test_image.csv', header=None)
    
# ## Reshape the matrices to make them fit to the model dimensions

# Xtrain = Xtrain / 255
# Xtest = Xtest / 255

# ## One Hot Encoding for Y set
# digits = 10
# xTrainShape = Xtrain.shape[0]
# Ytrain = Ytrain.values.reshape(1, xTrainShape)
# Ytrain2 = np.eye(digits)[Ytrain.astype('int32')]
# Ytrain2 = Ytrain2.T.reshape(digits, xTrainShape)

# Xtrain = Xtrain.T
# Xtest = Xtest.T


# In[5]:


## Read inputs from the file (as numpy arrays directly)

## X -> Images, Y -> Labels

n = len(sys.argv)

if n == 4:
    Xtrain = genfromtxt(sys.argv[1], delimiter=',')
    Ytrain = genfromtxt(sys.argv[2], delimiter=',')
    Xtest = genfromtxt(sys.argv[3], delimiter=',')
else:
    Xtrain = genfromtxt('train_image.csv', delimiter=',')
    Ytrain = genfromtxt('train_label.csv', delimiter=',')
    Xtest = genfromtxt('test_image.csv', delimiter=',')


# In[6]:


## Reshape the matrices to make them fit to the model dimensions

# ## Convert to Arrays
# Xtrain = np.asarray(Xtrain)
# Xtest = np.asarray(Xtest)

# ## Normalize the Data
# Xtrain = np.multiply(Xtrain, 1.0 / 255.0)
# Xtrain = Xtrain.astype('float32')


Xtrain = Xtrain / 255
Xtest = Xtest / 255

## One Hot Encoding for Y set
digits = 10
xTrainShape = Xtrain.shape[0]
xTestShape = Xtest.shape[0]

Ytrain = Ytrain.reshape(1, xTrainShape)
Ytrain2 = np.eye(digits)[Ytrain.astype('int32')]
Ytrain2 = Ytrain2.T.reshape(digits, xTrainShape)

Xtrain = Xtrain.T
Xtest = Xtest.T


# In[7]:


np.random.seed(138)

X = Xtrain
Y = Ytrain2


# In[8]:


## Declaring Hyperparameters

numX = Xtrain.shape[0]
numH1 = 512
numH2 = 64
batchSize = 64
batches = -(-xTrainShape // batchSize)
learningRate = 4
epochs = 20
beta = 0.9


# In[9]:


## Initializing the parameters

parameters = {'W1': np.random.randn(numH1, numX) * np.sqrt(1.0 / numX),
              'W2': np.random.randn(numH2, numH1) * np.sqrt(1.0 / numH1),
              'W3': np.random.randn(digits, numH2) * np.sqrt(1.0 / numH2),
              'b1': np.zeros((numH1, 1)) * np.sqrt(1.0 / numX),
              'b2': np.zeros((numH2, 1)) * np.sqrt(1.0 / numH1),
              'b3': np.zeros((digits, 1)) * np.sqrt(1.0 / numH2)}

valW1 = np.zeros(parameters["W1"].shape)
valW2 = np.zeros(parameters["W2"].shape)
valW3 = np.zeros(parameters["W3"].shape)
valb1 = np.zeros(parameters["b1"].shape)
valb2 = np.zeros(parameters["b2"].shape)
valb3 = np.zeros(parameters["b3"].shape)


# In[10]:


## Main Training of the model

for i in range(epochs):

    ## Mini batch gradient descent
    permutation = np.random.permutation(Xtrain.shape[1])
    XtrainShuffled = Xtrain[:, permutation]
    YtrainShuffled = Ytrain2[:, permutation]

    for j in range(batches):

        start = j * batchSize
        end = min(start + batchSize, Xtrain.shape[1] - 1)
        X = XtrainShuffled[:, start:end]
        Y = YtrainShuffled[:, start:end]
        mBatch = end - start

        output = feedForwardPass(X, parameters)
        gradient = backPropagation(X, Y, parameters, output)

        ## Momemtum Step
        valW1 = (beta * valW1 + (1.0 - beta) * gradient['gradW1'])
        valW2 = (beta * valW2 + (1.0 - beta) * gradient['gradW2'])
        valW3 = (beta * valW3 + (1.0 - beta) * gradient['gradW3'])
        valb1 = (beta * valb1 + (1.0 - beta) * gradient['gradb1'])
        valb2 = (beta * valb2 + (1.0 - beta) * gradient['gradb2'])
        valb3 = (beta * valb3 + (1.0 - beta) * gradient['gradb3'])

        ## Update weights/gradients
        parameters['W1'] = parameters['W1'] - learningRate * valW1
        parameters['W2'] = parameters['W2'] - learningRate * valW2
        parameters['W3'] = parameters['W3'] - learningRate * valW3
        parameters['b1'] = parameters['b1'] - learningRate * valb1
        parameters['b2'] = parameters['b2'] - learningRate * valb2
        parameters['b3'] = parameters['b3'] - learningRate * valb3

    output = feedForwardPass(Xtrain, parameters)
    cost = crossEntropyLoss(Ytrain2, output['A3'])
    # print("Epoch {}: training cost = {}".format(i+1, cost))


# In[11]:


## Prediction

cache = feedForwardPass(Xtest, parameters)
predictions = np.argmax(cache['A3'], axis = 0)


# In[12]:


## Saving predictions to file

np.savetxt("test_predictions.csv", predictions, delimiter=",", fmt="%d")


# In[13]:


# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Ytest = pd.read_csv('test_label.csv', header=None)

# # ## One Hot Encoding for Y Test set (for validation)
# # yshapeTest = Ytest.shape[0]
# # Ytest2 = Ytest.values.reshape(1, yshapeTest)
# # Ytest2 = np.eye(digits)[Ytest.astype('int32')]
# # Ytest2 = Ytest2.T.reshape(digits, yshapeTest)

# Ytest = genfromtxt('test_label.csv', delimiter=',')

# ## One Hot Encoding for Y Test set (for validation)
# Ytest = Ytest.reshape(1, xTestShape)
# Ytest2 = np.eye(digits)[Ytest.astype('int32')]
# Ytest2 = Ytest2.T.reshape(10, xTestShape)

# ## Check Accuracy

# labels = np.argmax(Ytest2, axis = 0)

# cm = confusion_matrix(predictions, labels)
# cr = classification_report(predictions, labels)

# print("Confusion Matrix: \n", cm)
# print()
# print("Classification Report: \n", cr)

# correct = 0
# for i in range(len(predictions)):
#     if predictions[i] == labels[i]:
#         correct = correct + 1
# print("No. of correct predictions: ", correct)

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax,cmap='Blues',fmt='');
# ax.set_xlabel('Predicted labels');
# ax.set_ylabel('True labels');
# ax.set_title('Confusion Matrix');
# ticklabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# ax.xaxis.set_ticklabels(ticklabels)
# ax.yaxis.set_ticklabels(ticklabels)


# In[14]:


## End recording time
endTime = time.time()

# print('Time Taken =', endTime - startTime)

# In[ ]:





