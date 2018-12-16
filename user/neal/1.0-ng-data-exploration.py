#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:17:25 2018

@author: nealgilmore
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import warnings

# ### Turn off Depreciation and Future warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# %reset -f

# Set the plotting style
print(plt.style.available)
plt.style.use('seaborn-whitegrid')

# Import the training set
filename = '0.0-sh-data-JPM.csv'
data = pd.read_csv(filename)
data.head()
data.shape

# ### Sort DataFrame by date
data = data.sort_values('Date')

# Double check the result
print(data.head())

# ### Visualize the data
# Plot all the data
print(data.plot(figsize=(12, 6)))

# Opening Price
plt.figure(figsize=(12, 6))
plt.plot(range(data.shape[0]), data['Open'])
plt.xticks(range(0, data.shape[0], 251), data['Date'].loc[::251], rotation=90)
plt.title('Daily Stock Price (Open): JPM')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Open Price (USD)', fontsize=14)

plt.savefig('1.0-ng-daily-stock-market-data-price-JPM-open.png',
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# Adjusted data
# Adj. Opening Price
plt.figure(figsize=(12, 6))
plt.plot(range(data.shape[0]), data['Adj_Open'])
plt.xticks(range(0, data.shape[0], 251), data['Date'].loc[::251], rotation=90)
plt.title('Daily Stock Price (Adj. Open): JPM')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Adj. Open Price (SUD)', fontsize=14)

plt.savefig('1.0-ng-daily-stock-market-data-price-JPM-adj-open.png',
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# ### Add Bollinger Bands
data['30 Day MA'] = data['Adj_Close'].rolling(window=20).mean()
data['30 Day STD'] = data['Adj_Close'].rolling(window=20).std()
data['Upper Band'] = data['30 Day MA'] + (data['30 Day STD'] * 2)
data['Lower Band'] = data['30 Day MA'] - (data['30 Day STD'] * 2)

# Simple 30 Day Bollinger Band for JPM
data[['Adj_Close', '30 Day MA', 'Upper Band', 'Lower Band']].plot(figsize=(12,6))
plt.title('30 Day Bollinger Band for JPM')
plt.xlabel('Day', fontsize=14)
plt.xticks(range(0, data.shape[0], 251), data['Date'].loc[::251], rotation=90)
plt.ylabel('Price (USD)', fontsize=14)

plt.savefig('1.0-ng-daily-stock-market-data-open-price-JPM-w-Bollinger.png',
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# ### Coloured Bollinger Bands
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

# Get index values for the X axis for JPM DataFrame
x_axis = data.index.get_level_values(0)

# Plot shaded 21 Day Bollinger Band for JPM
ax.fill_between(x_axis, data['Upper Band'], data['Lower Band'], color='grey')

# Plot Adjust Closing Price and Moving Averages
ax.plot(x_axis, data['Adj_Close'], color='#003366', lw=2)
ax.plot(x_axis, data['30 Day MA'], color='black', lw=2)

# Set Title & Show the Image
ax.set_title('30 Day Bollinger Band For JPM')
ax.set_xlabel('Day', fontsize=14)
plt.xticks(range(0, data.shape[0], 251), data['Date'].loc[::251], rotation=90)
ax.set_ylabel('Price (USD)', fontsize=14)
ax.legend()

plt.savefig('1.0-ng-daily-stock-market-data-open-price-JPM-w-Bollinger-filled.png',
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# Prepare data for LSTM RNN modelling
# NOTE: NORMALLY HAVE THIS AS PART OF MY ACTUAL MODEL CODE

# Determine prediction period
num_days_pred = 60

# Split data into Training and Test sets
data_train = data[:(len(data) - num_days_pred)]  # All stock data except last 60 days
data_test = data[-num_days_pred:] # Last 60 days of stock data

# Check the shape of the training and test data
print(data_train.shape)
print(data_test.shape)

# Describe the training and test data
print(data_train.describe().T)
print(data_test.describe().T)

# Create a numpy array of 1 column that we care about - Open Stock Price
training_set = data_train.iloc[:, 1:2].values

# Get the real Opening stock prices for last 60 days
real_stock_price = data_test.iloc[:, 1:2].values

# Feature Scaling
# With RNNs it is recommended to apply normalization for feature scaling
sc = MinMaxScaler(feature_range=(0, 1),
                  copy=True)

# Scale the training set
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output (use the previous
# 60 days' stock prices to predict the next output = 3 months of prices)
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
                      X_train.shape[1], # number of columns in x_train
                      1))  # number of input layers (currently only opening price)

print(X_train.view())

print("X_train ndim: ", X_train.ndim)
print("X_train shape:", X_train.shape)
print("X_train size: ", X_train.size)

print(y_train.view())

print("y_train ndim: ", y_train.ndim)
print("y_train shape:", y_train.shape)
print("y_train size: ", y_train.size)