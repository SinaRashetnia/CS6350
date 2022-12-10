#!/usr/bin/env python
# coding: utf-8

# In[151]:


import numpy as np
import time
import random


# In[152]:


def load_data():
    
    train = []
    train_labels = []
    with open("/Users/sinarashetnia/Desktop/bank-note/train.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            train.append(item[:-1])
            train_labels.append([int(item[-1])])
            
    test = []
    test_labels = []
    with open("/Users/sinarashetnia/Desktop/bank-note/test.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            test.append(item[:-1])
            test_labels.append([int(item[-1])])
            
    return np.asarray(train, dtype= float), np.asarray(train_labels, dtype= int), np.asarray(test, dtype= float), np.asarray(test_labels, dtype= int)


# In[153]:


X_train, y_train, X_test, y_test =  load_data()


# In[154]:


n_data = X_train.shape[0]
n_data


# In[177]:


class NN:
    
    def __init__(self, n_1, n_2):
        
        self.n_1 = n_1
        self.n_2 = n_2
        
        
        
    def fit(self, X_train, y_train, l_rate, n_iteration, batch_size = 1):
            
        self.X = X_train
        self.y = y_train
            
            
        self.l = l_rate
        self.N = X_train.shape[0]
        m = X_train.shape[1]
            
            
        self.W_0 =  1 * np.random.normal(0, 1, (m, self.n_1))
        self.W_1 =  1 * np.random.normal(0, 1, (self.n_1, self.n_2))
        self.W_2 =  1 * np.random.normal(0, 1, (self.n_2, 2))
      #  self.W_0 = 1* np.zeros((m, self.n_1))

     #   self.W_1 = 1* np.zeros((self.n_1, self.n_2))
     #   self.W_2 = 1* np.zeros((self.n_2, 2))

       
            
        self.b_0 = 0 
        self.b_1 = 0 
        self.b_2 = 0 
        
        
            
        for i in range(n_iteration):
                
            L = random.choices(np.arange(self.N), k = batch_size)
            X_bach = X_train[L, :]
            y_bach = y_train[L, :]
            
        
                
            self.f(X_bach)
            
            self.update(y_bach)
                        
                
                
            self.W_0 = self.W_0 - (1/batch_size) * self.l * self.dW_0 
            self.b_0 = self.b_0 - (1/batch_size) * self.l * self.db_0
                
                
            self.W_1 = self.W_1 - (1/batch_size) * self.l * self.dW_1
            self.b_1 = self.b_1 - (1/batch_size) * self.l * self.db_1
            
            self.W_2 = self.W_2 - (1/batch_size) * self.l * self.dW_2
            self.b_2 = self.b_2 - (1/batch_size) * self.l * self.db_2
            
                        
    def f(self, X):
        
        n_batch = X.shape[0]
                
        self.Z_0 = X
        self.S_0 = X
        
        self.S_1 = self.Z_0 @ self.W_0 + self.b_0
        self.Z_1 = np.maximum(0, self.S_1) 
        
        self.S_2 = self.Z_1 @ self.W_1 + self.b_1
        self.Z_2 = np.maximum(0, self.S_2) 
         
        self.scores = self.Z_2 @ self.W_2 + self.b_2
        

    def update(self, y):
                
        n = y.shape[0]
        
        exp_scores = np.exp(self.scores)
        
        
        softmax_matrix = exp_scores/np.sum(exp_scores, axis=1, keepdims = True)
        
        temp1 = softmax_matrix[np.arange(n), y.reshape(-1,)]
        
        self.softmax_loss = np.sum(-np.log(temp1))/n 
        
        softmax_matrix[np.arange(n), y.reshape(-1,)] -= 1 
        
        dS_3 = softmax_matrix
        
        
        
        
        self.dW_2 = self.Z_2.T @ dS_3
        self.db_2 = np.sum(dS_3, axis=0, keepdims = True)
        dZ_2 = dS_3 @ self.W_2.T 
        dS_2 = dZ_2 * (self.Z_2 >0)
        
        
        self.dW_1 = self.Z_1.T @ dS_2
        self.db_1 = np.sum(dS_2, axis=0, keepdims = True)
        dZ_1 = dS_2 @ self.W_1.T 
        dS_1 = dZ_1 * (self.Z_1 >0)
        
        self.dW_0 = self.Z_0.T @ dS_1
        self.db_0 = np.sum(dS_1, axis=0, keepdims = True)
                                   
    def predict(self, X):
        
        Z = np.maximum(0, X @ self.W_0 + self.b_0)
        Z = np.maximum(0, Z @ self.W_1 + self.b_1)
        Z = Z @ self.W_2 + self.b_2
        return np.argmax(Z, axis = 1)


# In[185]:


A = NN(5, 5)


# In[186]:


A.fit(X_train[:,:], y_train[:, :], l_rate = .01, n_iteration= 1000, batch_size = 1)


# In[187]:


1- np.sum(A.predict(X_test).reshape(-1,1) == y_test)/500


# In[188]:


1- np.sum(A.predict(X_train[:, :]).reshape(-1,1) == y_train[:, :])/X_train[:, :].shape[0]


# In[ ]:




