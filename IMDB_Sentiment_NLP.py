# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 13:17:13 2018

@author: sn06
"""

import os
import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import rmsprop
import matplotlib.pyplot as plt

def import_data():
    train_data=[]
    train_labels=[]
    test_data=[]
    test_labels=[]
    for i in ['train','test']:
        for j in ['pos','neg']:
            for k in os.listdir(os.getcwd() + '/' + i + '/' + j):
                filepath=os.getcwd() + '/' + i + '/' + j + '/' + k
                z = pd.read_csv(filepath,quoting=csv.QUOTE_NONE,header=None,sep='\n')
                if i=='train':
                    train_data.append(z.values)
                    if j=='pos':
                        train_labels.append(1)
                    elif j=='neg':
                        train_labels.append(0)
                if i=='test':
                    test_data.append(z.values)
                    if j=='pos':
                        test_labels.append(1)
                    elif j=='neg':
                        test_labels.append(0)
    train_data = np.array(train_data).reshape(-1)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data).reshape(-1)
    test_labels = np.array(test_labels)
    return train_data,train_labels,test_data,test_labels

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(input_dim,activation='relu',input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(int((input_dim+2)/2),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer=rmsprop(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
    return model

X_train,y_train,X_test,y_test = import_data()

tfidf = TfidfVectorizer(stop_words='english',min_df=40)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

model = create_model(X_train.shape[1])
history = model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))

f,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(history.history['loss'],label='loss')
ax1.plot(history.history['val_loss'],label='val_loss')
ax1.legend()
ax2.plot(history.history['acc'],label='acc')
ax2.plot(history.history['val_acc'],label='val_acc')
ax2.legend()
plt.tight_layout()
plt.show()