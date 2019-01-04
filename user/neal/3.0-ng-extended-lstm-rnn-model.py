#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:14:40 2019

Adapted from code created as part of Udemy course: Deep
Learning A-Z™: Hands-On Artificial Neural Networks
(https://www.udemy.com/deep-learning-a-z/)
"""

# # Deep Learning: Stacked LSTM Recurrent Neural Network (RNN) Model in Python
# Predict stock price using a Long-Short Term Memory (LSTM) Recurrent Neural
# Network (RNN)

# ## Part 1: Data Preprocessing
# Import required libraries
import os
import datetime as dt
import quandl
import pandas as pd
import numpy as np
import urllib.request
import json
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential  # Initializes the Neural Network
from keras.layers import Dense, Dropout, LSTM
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor

import config  # stores my API keys

import warnings

# ### Turn off Depreciation and Future warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# %reset -f

# Set random seed
seed = 42
np.random.seed(seed)

# Set the plotting style
plt.style.use('seaborn-whitegrid')

# Set the date range
now = dt.datetime.now()
file_date = now.strftime('%Y-%m-%d')

start_date = dt.datetime(now.year - 10, now.month, now.day)
start_date = start_date.strftime('%Y-%m-%d')
#end_date = dt.datetime(now.year, now.month, now.day)
#end_date = pd.to_datetime(end_date)

# Determine prediction period
num_days_pred = 80

# Set params
dropout_rate = 0.2
batch_size = 32
epochs = 500
