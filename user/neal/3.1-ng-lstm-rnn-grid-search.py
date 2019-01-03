#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Dec 31 12:35:23 2018
@author: ngilmore

Adapted from code created as part of Udemy course: Deep
Learning A-Z™: Hands-On Artificial Neural Networks
(https://www.udemy.com/deep-learning-a-z/)
"""

# ## Part 1: Data Preprocessing
# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential  # Initializes the Neural Network
from keras.layers import Dense, Dropout, LSTM
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor

# %matplotlib inline

# Determine prediction period
num_days_pred = 80

# Import the training set
data = pd.read_csv('../data/processed/0.0-sh-data-JPM.csv')
print(data.head())
print(data.shape)

# Split data into Training and Test sets
# All stock data except last 60 days
data_train = data[:(len(data) - num_days_pred)]
data_test = data[-num_days_pred:]  # Last n days of stock data

# Plot Training set
plt.figure(figsize=(12, 6))
plt.plot(range(data_train.shape[0]), data_train['Adj_Open'])
plt.xticks(range(0, data_train.shape[0], 251),
           data_train['Date'].loc[::251], rotation=90)
plt.title('Daily Stock Price (Adj. Open): JPM [Training Data]')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Adj. Open Price (USD)', fontsize=14)

plt.savefig('../reports/figures/3.1-ng-training-data-JPM-adj-open.png',
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# Plot Test set
plt.figure(figsize=(12, 6))
plt.plot(range(data_test.shape[0]), data_test['Adj_Open'])
plt.xticks(range(0, data_test.shape[0], num_days_pred - 1),
           data_test['Date'].loc[::num_days_pred - 1], rotation=90)
plt.title('Daily Stock Price (Adj. Open): JPM [Test Data]')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Adj. Open Price (USD)', fontsize=14)

plt.savefig('../reports/figures/3.1-ng-test-data-JPM-adj-open.png',
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# Describe the training data
print(data_train.shape)
print(data_train.describe().T)

# Describe the test data
print(data_test.shape)
print(data_test.describe().T)

# Create a numpy array of 1 column that we care about - Adj Open Stock Price
training_set = data_train.iloc[:, 8:9].values

# Get the real Adj Opening stock prices for last n days
real_stock_price = data_test.iloc[:, 8:9].values

# Feature Scaling
# With RNNs it is recommended to apply normalization for feature scaling
sc = MinMaxScaler(feature_range=(0, 1),
                  copy=True)

# Scale the training set
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with n timesteps and 1 output (use the previous
# n days' stock prices to predict the next output = n/20 months of prices)
X_train = []
y_train = []

for i in range(num_days_pred, len(data_train)):

    # append the previous 60 days' stock prices
    X_train.append(training_set_scaled[i - num_days_pred:i, 0])

    # predict the stock price on the next day
    y_train.append(training_set_scaled[i, 0])

# Convert X_train and y_train to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data to add additional indicators (e.g. volume, closing price, etc.)
# if needed (currently only predicting opening price)
X_train = np.reshape(X_train,
                     (X_train.shape[0],  # number of rows in x_train
                      X_train.shape[1],  # number of columns in x_train
                      1))  # number of input layers (currently only opening price)
