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
# set up api key 
#quandl.ApiConfig.api_key = "YOUR_KEY_HERE"

#------------------------------------------------------------------------------
# IMPORT FILES
#setup directories
inputdir = 'D:\QUEENS MMAI\894 Deep\Team Project\Input'
outputdir = 'D:\QUEENS MMAI\894 Deep\Team Project\Output'



# set dir
os.chdir(outputdir)
#------------------------------------------------------------------------------



mydata = quandl.get("FRED/GDP")


'''
NOTES 

Ill have to try proper scraping some other time



'''