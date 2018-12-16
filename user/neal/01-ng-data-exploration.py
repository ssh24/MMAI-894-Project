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
import pandas_datareader.data as web
import quandl
from pandas_datareader.data import Options
import urllib.request
import json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import config  # stores API keys

import warnings

# Turn off Depreciation and Future warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# %reset -f # reset environment if needed

# Set the plotting style
plt.style.use('bmh')

# Set the date range
now = dt.datetime.now()

start_date = dt.datetime(now.year - 10, now.month, now.day)
end_date = dt.datetime(now.year, now.month, now.day)

# Import a stock
stock_symbol = 'JPM'

##########
# # Quandl API
##########
quandl.ApiConfig.api_key = config.quandl_api_key
df_quandl = quandl.get('EOD/%s' % stock_symbol,
                       start_date=start_date, end_date=end_date)

# ### Sort DataFrame by date
#df = df.sort_values('Date')
print(df_quandl.columns.values)

print(df_quandl.head())
