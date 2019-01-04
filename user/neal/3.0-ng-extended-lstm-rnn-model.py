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

##########
# # Alpha Vantage API
##########
api_key = config.alpha_vantage_api_key

# Import list of stocks
# Use 'TSX:' to identify Canadian stocks
stock_symbols = ['AAPL', 'IBM', 'TSLA', 'RY', 'JPM']

# Loop through each stock in list
for index in range(0, len(stock_symbols)):

    stock_symbol = stock_symbols[index]

    print("\n*********** STOCK SYMBOL: %s ***********\n" % stock_symbol)

    url_string_daily = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=%s&outputsize=full&apikey=%s" % (
        stock_symbol, api_key)

    # Save data to file
    save_to_file = '../data/processed/3.0-ng-alpha-vantage-intraday-stock-market-data-%s-%s.csv' % (
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
    print('Data saved to : %s' % save_to_file)
    df_alpha.to_csv(save_to_file, index=False)

    # Load it from the CSV
    print('This file already exists. Loading data from CSV')
    df_alpha = pd.read_csv(save_to_file)

    # ### Sort DataFrame by date
    df_alpha = df_alpha.sort_values('Date')

    # Filter data to last n years from start_date
    df_alpha = df_alpha[(df_alpha['Date'] > start_date)]

    # Double check the result
#    print(df_alpha.head())

#    print(df_alpha.plot(figsize=(12, 6)))

    # ### Visualize the Adjusted Close Price
    plt.figure(figsize=(12, 6))
    plt.plot(range(df_alpha.shape[0]), df_alpha['Adjusted Close'])
    plt.xticks(range(0, df_alpha.shape[0], 251),
               df_alpha['Date'].loc[::251], rotation=90)
    plt.title('Daily Stock Price (Adj Close): %s' % stock_symbol)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Adj. Close Price', fontsize=14)

    plt.savefig('../reports/figures/3.0-ng-alpha-vantage-daily-stock-market-data-adj-close-price-%s.png' % stock_symbol,
                bbox_inches='tight',
                dpi=300)
    print(plt.show())

    # ### Splitting Data into Training and Test
    # Define price variables for training and testing purposes
#    open_prices = df_alpha.loc[:, 'Open'].values
#    close_prices = df_alpha.loc[:, 'Close'].values
#    adj_close_prices = df_alpha.loc[:, 'Adjusted Close'].values
#    high_prices = df_alpha.loc[:, 'High'].values
#    low_prices = df_alpha.loc[:, 'Low'].values
#    mid_prices = (high_prices + low_prices) / 2.0

    # ### NOTE: We can't use train_test_split because it would randomly pick rows and the
    # ### order of rows is critical to our analysis

    # Split data into Training and Test sets
    # All stock data except last 60 days
    data_train = df_alpha[:(len(df_alpha) - num_days_pred)]
    data_test = df_alpha[-num_days_pred:]  # Last n days of stock data
