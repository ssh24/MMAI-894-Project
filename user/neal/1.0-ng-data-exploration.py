#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:17:25 2018

@author: nealgilmore
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import quandl

import config  # stores my API keys

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