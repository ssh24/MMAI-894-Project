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
