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
