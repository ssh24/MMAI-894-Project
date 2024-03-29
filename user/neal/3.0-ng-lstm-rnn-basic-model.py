#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Dec 07 13:07:38 2018
@author: ngilmore

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils.vis_utils import plot_model

# %matplotlib inline

# Determine prediction period
num_days_pred = 80

# Import the training set
data = pd.read_csv('../data/processed/0.0-sh-data-JPM.csv')

data.describe().T

# Split data into Training and Test sets
# All stock data except last n days
data_train = data[:(len(data) - num_days_pred)]
data_test = data[-num_days_pred:]  # Last n days of stock data

# Create a numpy array of 1 column that we care about - Adj Open Stock Price
training_set = data_train.iloc[:, 8:9].values

# Get the real Opening stock prices for the last n days
real_stock_price = data_test.iloc[:, 8:9].values

# Feature Scaling
# With RNNs it is recommended to apply normalization for feature scaling
sc = MinMaxScaler(feature_range=(0, 1),
                  copy=True)

# Scale the training set
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with n timesteps and 1 output (use the previous
# n days' stock prices to predict the next output = 3 months of prices)
X_train = []
y_train = []

for i in range(num_days_pred, len(data_train)):

    # append the previous n days' stock prices
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

# Part 2: Build the Recurrent Neural Network (RNN) Model
# Import the required Keras libraries and packages

# Add a timer
start = timer()

# Set params
dropout_rate = 0.2
batch_size = 32
epochs = 500

# Initialize the RNN
model = Sequential()

# Add the 1st LSTM layer with Dropout regularization
model.add(LSTM(units=num_days_pred,  # number of memory cells (neurons) in this layer
               return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
model.add(Dropout(rate=dropout_rate))

# Add a 2nd LSTM layer with Dropout regularization
model.add(LSTM(units=num_days_pred,  # number of memory cells (neurons) in this layer
               return_sequences=True))
model.add(Dropout(rate=dropout_rate))

# Add a 3rd LSTM layer with Dropout regularization
model.add(LSTM(units=num_days_pred,  # number of memory cells (neurons) in this layer
               return_sequences=True))
model.add(Dropout(rate=dropout_rate))

# Add a 4th (and last) LSTM layer with Dropout regularization
# number of memory cells (neurons) in this layer
model.add(LSTM(units=num_days_pred))
model.add(Dropout(rate=dropout_rate))

# Add the output layer
model.add(Dense(units=1))

# Compile the Recurrent Neural Network (RNN)
model.compile(optimizer='adam',
              loss='mean_squared_error')

# Fit the Recurrent Neural Network (RNN) to the Training Set
model.fit(x=X_train,
          y=y_train,
          batch_size=batch_size,
          epochs=epochs)

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
os.system('say "your model has finished processing"')

# Print summary of the neural network architecture
print(model.summary())

# Part 3: Make Prediction and Visualize the Results
# Get the predicted Open stock prices for last n days
data_total = data['Adj_Open']

# first financial day is the difference between the length of the dataset_total and dataset_test
inputs = data_total[len(data_total) - len(data_test) - num_days_pred:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)  # Scale the inputs

X_test = []

for i in range(num_days_pred, len(data_test) + num_days_pred):
    # append the previous n days' stock prices
    X_test.append(inputs[i-num_days_pred:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test,
                    (X_test.shape[0],
                     X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)

# Invert the feature scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Set the plotting style
plt.style.use('seaborn-whitegrid')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(real_stock_price,
         color='red',
         label='Real JPM Stock Price (Last %s Days)' % num_days_pred)
plt.plot(predicted_stock_price,
         color='blue',
         label='Predicted JPM Stock Price (Last %s Days)' % num_days_pred)
plt.title('JPM Stock Price Prediction (%s Days : %s Epochs)' % (num_days_pred, epochs))
plt.xticks(range(0, data_test.shape[0], 5), data_test['Date'].loc[::5], rotation=90)
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()

plt.savefig('../reports/figures/3.0-ng-basic-lstm-rnn-model-last-%s-days-%s-epochs.png' % (num_days_pred, epochs),
            bbox_inches='tight',
            dpi=300)
print(plt.show())
