# import libraries for Investors Exchange (IEX)

from pandas_datareader import data as web
from datetime import datetime

start = datetime(2015, 2, 9)
end = datetime(2017, 5, 24)

f = web.DataReader('MSFT', 'iex', start, end)

print(f.head())
