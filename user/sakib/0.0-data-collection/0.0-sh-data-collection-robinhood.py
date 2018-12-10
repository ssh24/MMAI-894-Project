# import libraries for Robinhood
# robinhood: a stock trading platform with an API that provides a limited set of data. Historical daily data is limited to 1 year relative to today.

from pandas_datareader import data as web

f = web.DataReader('MSFT', 'robinhood')

print(f.columns)

print(f.head())

print(f.tail())
