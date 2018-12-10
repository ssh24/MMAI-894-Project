# import standard libraries
import os

# import libraries required for reading data from the web
from pandas_datareader import data as web
from datetime import date

# get the current working directory
cwd = os.getcwd()

# set a start date of 1970 Jan 1st to fetch the stock data from
start = date(1970, 1, 1)
# grab the latest date for the end date
end = date.today()

# set api to be the end of day stock price
api = "EOD/"
# set the source to be quandl
source = "quandl"

# list of companies we want to grab stock for
stocks = ['MSFT', 'IBM', 'AAPL', 'INTC', 'JPM', 'AXP']

# loop through each company, grab the data, display some information and save it in a csv file
for i in range(0, len(stocks)):
    stock = api + stocks[i]

    df = web.DataReader(stock, source, start, end)
    df.reset_index(inplace=True)

    print "\n\n****** Printing data information for: " + stocks[i] + " ******"

    print "\nShape: ", df.shape

    print "\nColumns: ", list(df.columns)

    print "\nCheck for null columns: ", df.isnull().any()

    print "\nGet the head: ", df.head()

    print "\nGet the tail: ", df.tail()

    print "\nWrite the data to csv ..."
    df.to_csv(os.path.join(cwd, 'data', '0.0-sh-data-' + stocks[i] + '.csv'), index=False)
    