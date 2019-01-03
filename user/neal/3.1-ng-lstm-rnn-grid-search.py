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
