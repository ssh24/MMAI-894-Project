# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 23:36:07 2018

lets do some basic data api stuff 

Alpha vantage and Quandl sound pretty damn good  

@author: 9atg
"""

#import standard packages
import os
import datetime
import random as rd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import other packages 
import quandl

#------------------------------------------------------------------------------
# set up quandl api key 
quandl.ApiConfig.api_key = "RcoFDV2VLyhEeYcwjQwW"

#------------------------------------------------------------------------------
# IMPORT FILES
#setup directories
inputdir = 'D:\QUEENS MMAI\894 Deep\Team Project\Input'
outputdir = 'D:\QUEENS MMAI\894 Deep\Team Project\Output'



# set dir
os.chdir(outputdir)
#------------------------------------------------------------------------------

data = quandl.get("EIA/PET_RWTC_D", start_date="2001-12-31", end_date="2005-12-31",collapse="monthly")



data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'close'] }, ticker = ['AAPL', 'MSFT'], date = { 'gte': '2016-01-01', 'lte': '2016-12-31' })

data = quandl.get_table('MER/F1', paginate=True, ticker = ['AAPL', 'MSFT'])

'''
NOTES 

I have no real care about what the data is... don't really know enough
Ill have to try proper scraping some other time

'''