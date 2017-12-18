
# coding: utf-8

# In[4]:

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn import datasets, metrics, preprocessing
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

#Load the dataset
df = pd.read_csv("data/CHD.csv", header=0)
#Describe the input data
print (df.describe())

#Normalize the input data
a = preprocessing.StandardScaler()
X =a.fit_transform(df['age'].reshape(-1, 1))

#Shuffle the data 
x,y = shuffle(X, df['chd'])

#Define the model as a logistic regression with
model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model with the first 90 elements, and spliting 70%/30% of them for training/validation sets

model.fit(x[:90], y[:90], nb_epoch=100, validation_split=0.33, shuffle=True,verbose=2 )

#Evaluate the model with the last 10 elements
scores = model.evaluate(x[90:], y[90:], verbose=2)
print (model.metrics_names)
print (scores)


# In[ ]:



