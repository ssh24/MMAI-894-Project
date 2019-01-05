
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
train = data[:(len(data)- window)]
test = data[-window:]


# In[16]:


#Create Array for target feature
train = train.iloc[:, 5:6].values


# In[17]:


#Pull out test Adj_Close prices for last 80 days 
test = test.iloc[:, 5:6].values


# In[18]:


#Normalise and scale data

scale = MinMaxScaler(feature_range=(0,1),copy=True)


# In[19]:


#Train Scaler with training data and smooth
#Code Adapted from https://www.datacamp.com/community/tutorials/lstm-python-stock-market
smoothing_window = 10
for i in range (0,2400, smoothing_window):
    scale.fit(train[i:i+smoothing_window,:])
    train[i:i+smoothing_window,:]=scale.transform(train[i:i+smoothing_window,:])
    
#normalise remaining data
scale.fit(train[i+smoothing_window:,:])
train[i+smoothing_window:,:] = scale.transform(train[i+smoothing_window:,:])


# In[20]:


train.shape


# In[21]:


print(train)


# In[22]:


test.shape


# In[23]:


#Smooth out training data using exponential moving average transformation
EMA = 0.0
gamma = 0.1
for ti in range (2437): 
    EMA = gamma*train[ti] + (1-gamma)*EMA
    train[ti] = EMA 
    
#For visualisations and testing
all_data = np.concatenate([train, test], axis=0)


# In[24]:


all_data.shape


# In[25]:


train.shape


# In[26]:


print(train)


# In[27]:


#split training into samples 
#adpated from https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/ 
samples =  list()
length = 250

for i in range(0,2250, length):
    sample = train[i:i+length]
    samples.append(sample)
print(len(samples))


# In[28]:


print(samples)


# In[29]:


#convert train to 2d array
#expecting (10,250)
train = array(samples).reshape(9,250)
print(train.shape)


# In[30]:


#Reshape train into samples, timesteps, features 
#expecting (10,250,1)
train = train.reshape((len(samples), length,1))


# In[31]:


#reshape train into [samples, timesteps, features]
train = train.reshape(9,length,1)
print(train.shape)


# In[32]:


#reshape test into [samples, timesteps, features]
test = test.reshape(80,1,1)


# In[33]:


#Build RNN Model
#adapted from Neal :) 


# In[42]:


#set params
dropout = 0.25
batch_size = 9
epochs = 100


# In[43]:


#MODEL


# In[44]:


#Initialise Model 
model = Sequential()


# In[45]:


#1st layer 
model.add(LSTM(units=window, return_sequences=True, input_shape=(250,1)))
model.add(Dropout(rate=dropout)) 


# In[46]:


#2nd layer 
model.add(LSTM(units=window, return_sequences=True))
model.add(Dropout(rate=dropout)) 


# In[48]:


#Output layer 
model.add(Dense(3)) 


# In[49]:


#Compile Model 
model.compile(optimizer='adam',loss='mean_squared_error') 


# In[41]:


#fit network to training set
model.fit(train, test, batch_size=batch_size, epochs=epochs)


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

all_data = data['Adj_Close']

