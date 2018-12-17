# import libraries
import numpy as np
import os
import pandas as pd

# environment settings
cwd = os.getcwd()

def main():
    # set working directory
    setwd(os.path.join(cwd, 'data'))

    # load initial data
    data = load_data('0.0-sh-data-JPM.csv')

    # get the shape of the data
    print "Shape of the data: ", get_shape(data)

    # get the list of columns
    print "List of columns: ", get_columns(data)

# set the current working directory
def setwd(wd):
    os.chdir(wd)

# load the dataframe
def load_data(file_name):
    return pd.read_csv(file_name)

# return the shape of the dataframe
def get_shape(data):
    return data.shape

def get_columns(data):
    return list(data.columns)

if __name__ == '__main__':
    main()
