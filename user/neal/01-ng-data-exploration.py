#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Dec 16 10:44:32 2018

@author: nealgilmore
"""

# Import required libraries
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import quandl


# import config  # stores API keys

import warnings

# Turn off Depreciation and Future warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# %reset -f # reset environment if needed

# Set the plotting style
plt.style.use('seaborn-whitegrid')

# Set the date range
now = dt.datetime.now()

start_date = dt.datetime(now.year - 10, now.month, now.day)
end_date = dt.datetime(now.year, now.month, now.day)

# Import a stock
stock_symbol = 'JPM'

##########
# # Quandl API
##########
quandl.ApiConfig.api_key = 'DaRhFhyMvSAHvZQ6uvn-'
df_quandl = quandl.get('EOD/%s' % stock_symbol,
                       start_date=start_date, end_date=end_date)

# ### Sort DataFrame by date
#df = df.sort_values('Date')
print(df_quandl.columns.values)

print(df_quandl.head())

# Save data to file
#save_to_file = '/data/processed/1.0-ng-quandl-daily-stock-market-data-%s.csv' % stock_symbol
#
#print(save_to_file)
#
#if not os.path.exists(save_to_file):
#    print('Data saved to : %s' % save_to_file)
#    df_quandl.to_csv(save_to_file, index=False)
#
## If the data is already there, just load it from the CSV
#else:
#    print('This file already exists. Loading data from CSV')
#    df_quandl = pd.read_csv(save_to_file)

# ### Sort DataFrame by date
df_quandl = df_quandl.sort_values('Date')

# Double check the result
print(df_quandl.head())

print(df_quandl.plot(figsize=(12, 6)))

# ### Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(range(df_quandl.shape[0]),
         (df_quandl['Adj_Low'] + df_quandl['Adj_High']) / 2.0)
plt.xticks(range(0, df_quandl.shape[0], 252),
           df_quandl['Date'].loc[::252], rotation=90)
plt.title('Daily Mid Stock Price: %s' % stock_symbol)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Mid Price', fontsize=14)


plt.savefig('/reports/figures/0.0-ng-quandl-daily-stock-market-data-price-%s-mid.png' % stock_symbol,
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# Opening Price
plt.figure(figsize=(12, 6))
plt.plot(range(df_quandl.shape[0]), df_quandl['Adj_Open'])
plt.xticks(range(0, df_quandl.shape[0], 252),
           df_quandl['Date'].loc[::252], rotation=90)
plt.title('Daily Stock Price (Open): %s' % stock_symbol)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Open Price', fontsize=14)


plt.savefig('/reports/figures/0.0-ng-quandl-daily-stock-market-data-price-%s-open.png' % stock_symbol,
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# Closing Price
plt.figure(figsize=(12, 6))
plt.plot(range(df_quandl.shape[0]), df_quandl['Adj_Close'])
plt.xticks(range(0, df_quandl.shape[0], 252),
           df_quandl['Date'].loc[::252], rotation=90)
plt.title('Daily Stock Price (Close): %s' % stock_symbol)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price', fontsize=14)


plt.savefig('/reports/figures/0.0-ng-quandl-daily-stock-market-data-price-%s-close.png' % stock_symbol,
            bbox_inches='tight',
            dpi=300)
print(plt.show())

# Volume
plt.figure(figsize=(12, 6))
plt.plot(range(df_quandl.shape[0]), df_quandl['Adj_Volume'])
plt.xticks(range(0, df_quandl.shape[0], 252),
           df_quandl['Date'].loc[::252], rotation=90)
plt.title('Daily Volume: %s' % stock_symbol)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Volume', fontsize=14)


plt.savefig('/reports/figures/0.0-ng-quandl-daily-stock-market-data-volume-%s.png' % stock_symbol,
            bbox_inches='tight',
            dpi=300)
print(plt.show())
