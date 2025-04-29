#!/usr/bin/env python
# coding: utf-8

# ## now we discussed trade forex.....

# In[7]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[8]:


data = pd.read_csv('all_stocks_5yr.csv')
data


# In[9]:


data.shape


# In[10]:


data.isnull().sum()


# In[11]:


dataset=data.dropna()
dataset


# In[12]:


dataset.shape


# In[13]:


dataset.duplicated().sum()


# In[14]:


dataset.count()


# In[15]:


dataset.info()


# In[16]:


dataset.nunique()


# In[17]:


dataset.describe()


# In[18]:


sns.pairplot(dataset)


# In[19]:


X=dataset.iloc[:,[1,2,3,5]].values
X


# In[20]:


Y= dataset.iloc[:,[4]].values
Y


# In[21]:


scaler=StandardScaler()

x_scaled= scaler.fit_transform(X)
print('The X values is now:')
print(x_scaled)

print()
print('/////////////////////////////////////////////')
y_scaled=scaler.fit_transform(Y)
print('The Y values is now')
print(y_scaled)


# In[22]:


X_train,X_test, Y_train,Y_test= train_test_split(x_scaled,y_scaled, test_size=0.2, random_state=42)


# In[23]:


X_train.shape


# In[24]:


X_test.shape


# In[25]:


Y_train.shape


# In[26]:


Y_test.shape


# In[28]:


model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  


# In[37]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

loss = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")

predictions = model.predict(X_test)
print(predictions)


# In[44]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('all_stocks_5yr.csv')

dataset = data.dropna()

X = dataset.iloc[:, [1, 2, 3, 5]].values
Y = dataset.iloc[:, [4]].values

# standardize the data......
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(Y)


X_train, X_test, Y_train, Y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear')) 

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)




# In[43]:


# Plot the actual vs predicted values using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(len(Y_test)), y=Y_test.flatten(), label='Actual', color='blue')
sns.scatterplot(x=range(len(predictions)), y=predictions.flatten(), label='Predicted', color='red')

# Add labels and title
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price (Scaled)')
plt.legend()
plt.show()


# In[47]:


# Plot the actual vs predicted values using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(len(Y_test[:100])), y=Y_test[:100].flatten(), label='Actual', color='blue')
sns.scatterplot(x=range(len(predictions[:100])), y=predictions[:100].flatten(), label='Predicted', color='red')

# Add labels and title
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price (Scaled)')
plt.legend()
plt.show()


# In[ ]:




