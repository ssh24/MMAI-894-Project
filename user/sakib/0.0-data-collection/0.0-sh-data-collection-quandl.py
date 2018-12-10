# import standard libraries
import os

# import libraries required for reading data from the web
import quandl
from datetime import date

# get the current working directory
cwd = os.getcwd()

# set your api key
API_KEY = os.environ['QUANDL_API_KEY']

# set a start date of 1970 Jan 1st to fetch the stock data from
start = date(2008, 12, 9)
# grab the latest date for the end date
end = date(2018, 12, 9)

# set api to be the end of day stock price
api = "EOD/"
# set the source to be quandl
source = "quandl"

# list of companies we want to grab stock for
stocks = ['MSFT', 'IBM', 'AAPL', 'INTC', 'JPM', 'AXP']

# loop through each company, grab the data, display some information and save it in a csv file
for id in stocks:
    stock = api + id

    df = quandl.get(stock, start_date=start, end_date=end, api_key=API_KEY)
    df.reset_index(inplace=True)

    print "\n\n****** Printing data information for: " + id + " ******"

    print "\nShape: ", df.shape

    print "\nColumns: ", list(df.columns)

    print "\nCheck for null columns: ", df.isnull().any()

    print "\nGet the head: ", df.head()

    print "\nGet the tail: ", df.tail()

    print "\nWrite the data to csv ..."
    df.to_csv(os.path.join(cwd, 'data', '0.0-sh-data-' + id + '.csv'), index=False)
