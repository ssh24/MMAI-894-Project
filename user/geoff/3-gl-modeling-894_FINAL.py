
# coding: utf-8

# In[1]:


#importing required packages
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
import datetime as dt
import os
import numpy as np
import tensorflow as tf 
import keras 

from timeit import default_timer as timer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from numpy import array 


# In[2]:


window = 80 


# In[3]:


#load dataset
JPM_data = pd.read_csv('/Users/Geoff/Downloads/0.0-sh-data-JPM (1).csv')


# In[4]:


#inspect the data
JPM_data.head()


# In[5]:


# inspect data shape
JPM_data.shape


# In[6]:


#describe data contents 
JPM_data.describe().T


# In[7]:


#check correlations
corr = JPM_data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(JPM_data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(JPM_data.columns)
ax.set_yticklabels(JPM_data.columns)
plt.show()


# In[8]:


#Drop uneccessary features 
data = JPM_data.drop([
            'Dividend',
            'Open',
            'High',
            'Low',
            'Close',                        
            'Volume',],
            axis=1)


# In[9]:


data = data.sort_values('Date')


# In[10]:


#Double Check Result (header)
data.head()


# In[11]:


#check data types
data.dtypes


# In[12]:


#Double Check Result (correlation)
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(1,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()


# In[13]:


#Visualise the Data (Adj_Close)
plt.figure(figsize=(12, 6))
plt.plot(range(data.shape[0]), data['Adj_Close'])
plt.xticks(range(0, data.shape[0], 251), data['Date'].loc[::251], rotation=90)
plt.title('Daily Stock Price (Adj. Open): JPM')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Adj. Close Price (SUD)', fontsize=14)

print(plt.show())


# In[14]:


#Add Momementum Feature


# In[15]:


#Split data into training and test sets ()
#Adapted from Neal :) 
train_len = data[:(len(data)- window)]
test = data[-window:]


# In[16]:


real_prices = test.iloc[:, 5:6].values
real_prices.shape


# In[17]:


#Create Array for target feature
train = train_len.iloc[:, 5:6].values


# In[18]:


#Pull out test Adj_Close prices for last 80 days 
test = test.iloc[:, 5:6].values


# In[19]:


#Normalise and scale data

scale = MinMaxScaler(feature_range=(0,1),copy=True)


# In[20]:


test.shape


# In[21]:


#Train Scaler with training data and smooth
#Code Adapted from https://www.datacamp.com/community/tutorials/lstm-python-stock-market
smoothing_window = 10
for i in range (0,2400, smoothing_window):
    scale.fit(train[i:i+smoothing_window,:])
    train[i:i+smoothing_window,:]=scale.transform(train[i:i+smoothing_window,:])
    
#normalise remaining data
scale.fit(train[i+smoothing_window:,:])
train[i+smoothing_window:,:] = scale.transform(train[i+smoothing_window:,:])


# In[22]:


train.shape


# In[23]:


print(train)


# In[24]:


test.shape


# In[25]:


#Smooth out training data using exponential moving average transformation
EMA = 0.0
gamma = 0.1
for ti in range (2437): 
    EMA = gamma*train[ti] + (1-gamma)*EMA
    train[ti] = EMA 


# In[26]:


train.shape


# In[27]:


print(train)


# In[28]:


# Create a data structure with n timesteps and 1 output (use the previous
# n days' stock prices to predict the next output = 3 months of prices)
#Adapted from Neal
X_train = []
y_train = []
window = 80

for i in range(window, len(train_len)):

    # append the previous n days' stock prices
    X_train.append(train[i - window:i, 0])

    # predict the stock price on the next day
    y_train.append(train[i, 0])

# Convert X_train and y_train to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data to add additional indicators (e.g. volume, closing price, etc.)
# if needed (currently only predicting opening price)
X_train = np.reshape(X_train,
                     (X_train.shape[0],  # number of rows in x_train
                      X_train.shape[1],  # number of columns in x_train
                      1))  # number of input layers (currently only opening price)


# In[29]:


#Build RNN Model
#adapted from Neal :) 


# In[30]:


#set params
dropout = 0.25
batch_size = 32
epochs = 1
input_dim = X_train.shape[1]


# In[31]:


#MODEL


# In[32]:


#Initialise Model 
model = Sequential()


# In[33]:


#1st layer 
model.add(LSTM(units=window,  # number of memory cells (neurons) in this layer
                   return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))
model.add(Dropout(rate=dropout))


# In[34]:


#2nd (and final layer) layer 
model.add(LSTM(units=window))
model.add(Dropout(rate=dropout)) 


# In[35]:


#Output layer 
model.add(Dense(units=1)) 


# In[36]:


#Compile Model 
model.compile(optimizer='adam',loss='mean_squared_error') 


# In[ ]:


#fit network to training set
model.fit(x=X_train,y=y_train, batch_size=batch_size,epochs=epochs)


# In[ ]:


# Elapsed time in minutes
#Adapted from Neal :) 
end = timer()
print('Minutes passed: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
os.system('say "model has finished processing"')

# Print summary of the neural network architecture
print(model.summary())


# In[ ]:


#Predict Stock Price 
#Adapted from Neal 

data_all = data['Adj_Close']

# first financial day is the difference between the length of the dataset_total and dataset_test
inputs = data_all[len(data_all) - len(test) - window:].values
inputs = inputs.reshape(-1, 1)
inputs = scale.transform(inputs)  # Scale the inputs

X_test = []

for i in range(window, len(test) + window):
    # append the previous n days' stock prices
    X_test.append(inputs[i-window:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test,
                    (X_test.shape[0],
                     X_test.shape[1], 1))

X_test.shape

X_test = array(X_test).reshape(-1,1)
X_test = X_test.reshape(80,1,1)
X_test.shape

pred_price = model.predict(X_test)


# In[ ]:


# Invert the feature scaling
pred_price = scale.inverse_transform(pred_price)

# Set the plotting style
plt.style.use('seaborn-whitegrid')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(real_prices,
         color='red',
         label='Real JPM Stock Price (Last %s Days)' % window)
plt.plot(pred_price,
         color='blue',
         label='Predicted JPM Stock Price (Last %s Days)' % window)
plt.title('JPM Stock Price Prediction (%s Days : %s Epochs)' % (window, epochs))
plt.xticks(range(0, test.shape[0], 5), test['Date'].loc[::5], rotation=90)
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()

