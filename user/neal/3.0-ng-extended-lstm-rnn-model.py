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
import pandas as pd
import numpy as np
import urllib.request
import json
import math
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential  # Initializes the Neural Network
from keras.layers import Dense, Dropout, LSTM

import config as cfg  # stores my API keys

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
# end_date = dt.datetime(now.year, now.month, now.day)
# end_date = pd.to_datetime(end_date)

# Determine prediction period
num_days_pred = 80

# Set params
dropout_rate = 0.2
batch_size = 32
epochs = 500

##########
# # Alpha Vantage API
##########
api_key = cfg.ALPHA_VANTAGE_API_KEY

# Import list of stocks
# Use 'TSX:' to identify Canadian stocks
stock_symbols = ['AAPL', 'IBM', 'TSLA', 'RY', 'JPM']

# Loop through each stock in list
for index in range(0, len(stock_symbols)):

    stock_symbol = stock_symbols[index]

    print("\n*********** STOCK SYMBOL: %s ***********\n" % stock_symbol)

#    url_string_daily = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=%s&outputsize=full&apikey=%s" % (
#        stock_symbol, api_key)
    url_string_daily = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize=full&apikey={}".format(
        stock_symbol, api_key)

    # Save data to file
    save_data_to_file = '../data/processed/3.0-ng-alpha-vantage-intraday-stock-market-data-{}-{}.csv'.format(
        stock_symbol, file_date)
    save_results_to_file = '../data/processed/3.0-ng-lstm-rnn-model-prediction-results-{}-{}.csv'.format(
        stock_symbol, file_date)

    # Store date, open, high, low, close, adjusted close, volume, and dividend amount values
    # to a Pandas DataFrame
    with urllib.request.urlopen(url_string_daily) as url:
        data = json.loads(url.read().decode())
        # extract stock market data
        data = data['Time Series (Daily)']
        df_alpha = pd.DataFrame(columns=['Date',
                                         'Open',
                                         'High',
                                         'Low',
                                         'Close',
                                         'Adjusted Close',
                                         'Volume',
                                         'Dividend Amount'])

        for k, v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(),
                        float(v['1. open']),
                        float(v['2. high']),
                        float(v['3. low']),
                        float(v['4. close']),
                        float(v['5. adjusted close']),
                        float(v['6. volume']),
                        float(v['7. dividend amount'])]
            df_alpha.loc[-1, :] = data_row
            df_alpha.index = df_alpha.index + 1
    print('Data saved to : {}'.format(save_data_to_file))
    df_alpha.to_csv(save_data_to_file, index=False)

    # Load it from the CSV
    df_alpha = pd.read_csv(save_data_to_file)

    # ### Sort DataFrame by date
    df_alpha = df_alpha.sort_values('Date')

    # Filter data to last n years from start_date
    df_alpha = df_alpha[(df_alpha['Date'] >= start_date)]

    # ### Visualize the Adjusted Close Price
    plt.figure(figsize=(12, 6))
    plt.plot(range(df_alpha.shape[0]), df_alpha['Adjusted Close'])
    plt.xticks(range(0, df_alpha.shape[0], 251),
               df_alpha['Date'].loc[::251], rotation=90)
    plt.title('Daily Stock Price (Adj Close): {}'.format(stock_symbol))
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Adj. Close Price', fontsize=14)

    plt.savefig('../reports/figures/3.0-ng-alpha-vantage-daily-stock-market-data-adj-close-price-{}-{}.png'.format(stock_symbol, file_date),
                bbox_inches='tight',
                dpi=300)
    print(plt.show())

    # ### Splitting Data into Training and Test
    # ### NOTE: We can't use train_test_split because it would randomly pick rows and the
    # ### order of rows is critical to our analysis

    # Split data into Training and Test sets
    # All stock data except last 60 days
    data_train = df_alpha[:(len(df_alpha) - num_days_pred)]
    data_test = df_alpha[-num_days_pred:]  # Last n days of stock data

    # Plot Training set
    plt.figure(figsize=(12, 6))
    plt.plot(range(data_train.shape[0]), data_train['Adjusted Close'])
    plt.xticks(range(0, data_train.shape[0], 251),
               data_train['Date'].loc[::251], rotation=90)
    plt.title(
        'Daily Stock Price (Adj. Close): {} [Training Data]'.format(stock_symbol))
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Adj. Close Price', fontsize=14)

    plt.savefig('../reports/figures/3.0-ng-training-data-{}-adj-close-{}.png'.format(stock_symbol, file_date),
                bbox_inches='tight',
                dpi=300)
    print(plt.show())

    # Plot Test set
    plt.figure(figsize=(12, 6))
    plt.plot(range(data_test.shape[0]), data_test['Adjusted Close'])
    plt.xticks(range(0, data_test.shape[0], 5),
               data_test['Date'].loc[::5], rotation=90)
    plt.title(
        'Daily Stock Price (Adj. Close): {} [Test Data]'.format(stock_symbol))
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Adj. Close Price', fontsize=14)

    plt.savefig('../reports/figures/3.0-ng-test-data-{}-adj-close-{}.png'.format(stock_symbol, file_date),
                bbox_inches='tight',
                dpi=300)
    print(plt.show())

#    # Describe the training data
#    print(data_train.shape)
#    print(data_train.describe().T)
#
#    # Describe the test data
#    print(data_test.shape)
#    print(data_test.describe().T)

    # Create a numpy array of 1 column that we care about - Adj Close Stock Price
    training_set = data_train.iloc[:, 5:6].values

    # Get the real Adj Closing stock prices for last n days
    real_stock_price = data_test.iloc[:, 5:6].values

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
              epochs=epochs,
              verbose=0)

    # Elapsed time in minutes
    end = timer()
    print('Elapsed time in minutes: ')
    print(0.1 * round((end - start) / 6))

    # Add an end of work message
    os.system('say "your {} model has finished processing"'.format(stock_symbol))

    # Print summary of the neural network architecture
    print(model.summary())

    # Part 3: Make Prediction and Visualize the Results
    # Get the predicted Open stock prices for last n days
    data_total = df_alpha['Adjusted Close']

    # first financial day is the difference between the length of the dataset_total and dataset_test
    inputs = data_total[len(data_total) -
                        len(data_test) - num_days_pred:].values
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

    # Add calculation of differences between prediction and actual - ups and downs
    df_date = pd.DataFrame(data_test.iloc[:, 0:1].values)
    df_real = pd.DataFrame(real_stock_price)
    df_pred = pd.DataFrame(predicted_stock_price)

    df_final = pd.concat([df_date, df_real, df_pred], axis=1)
    df_final.columns = ['Date', 'Real Price', 'Predicted Price']
    df_final['Real Change'] = df_final['Real Price'] - \
        df_final['Real Price'].shift(1)
    df_final['Real Change'].fillna(0, inplace=True)
    df_final['Real Direction'] = np.where(
        df_final['Real Change'] >= 0, 'Up', 'Down')
    df_final['Predicted Change'] = df_final['Predicted Price'] - \
        df_final['Predicted Price'].shift(1)
    df_final['Predicted Change'].fillna(0, inplace=True)
    df_final['Predicted Direction'] = np.where(
        df_final['Predicted Change'] >= 0, 'Up', 'Down')
    df_final['Correct Prediction'] = np.where(
        (df_final['Real Change'] * df_final['Predicted Change']) >= 0, True, False)

    df_final.to_csv(save_results_to_file, index=False)

    pred_acc = round(df_final['Correct Prediction'].sum(
    ) / len(df_final['Correct Prediction']) * 100, 2)

    print('\nPrediction Accuracy: {}%'.format(pred_acc))

    # Calculate mean squared error
    test_MSE = mean_squared_error(
        real_stock_price, predicted_stock_price, multioutput='raw_values')

    # Calculate root mean squared error
    test_RMSE = math.sqrt(mean_squared_error(
        real_stock_price, predicted_stock_price))

    print('\nTest Score: %.2f MSE' % (test_MSE))
    print('Test Score: %.2f RMSE' % (test_RMSE))

    # Visualize the results
    # Visualize only predicted period
    plt.figure(figsize=(12, 6))
    plt.plot(real_stock_price,
             color='red',
             label='Real {} Stock Price (Last {} Days)'.format(stock_symbol, num_days_pred))
    plt.plot(predicted_stock_price,
             color='blue',
             label='Predicted {} Stock Price (Last {} Days)'.format(stock_symbol, num_days_pred))
    plt.title('{} Stock Price Prediction ({} Days : {} Epochs)'.format(
        stock_symbol, num_days_pred, epochs))
    plt.xticks(range(0, data_test.shape[0], 5),
               data_test['Date'].loc[::5], rotation=90)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()

    plt.savefig('../reports/figures/3.0-ng-lstm-rnn-extended-model-{}-last-{}-days-{}-epochs-{}.png'.format(stock_symbol, num_days_pred, epochs, file_date),
                bbox_inches='tight',
                dpi=300)
    print(plt.show())
