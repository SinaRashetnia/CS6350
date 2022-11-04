#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import random 


# # Standard perceptron

# In[95]:


train = []
with open ("/Users/sinarashetnia/Desktop/bank-note/train.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a = [1] + list(map(float,a))
        train.append(a)

train = np.array(train)


# In[96]:


test = []
with open ("/Users/sinarashetnia/Desktop/bank-note/test.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a =[1] + list(map(float,a))
        test.append(a)

test = np.array(test)


# In[97]:


def perceptron(D, r, T):
    """
    X: train data; np.array
    y: train labels; np.array
    T: epoch; int
    """
    w = np.zeros(len(D[0]) - 1)  # e.g. 6 - 1 = 5
    n = len(D) # 872

    for t in range(T):
        np.random.shuffle(D)
        X = D[:, :-1] # train
        y = 2 * D[:, -1] - 1 # labels
        for i in range(n):
            if y[i] * np.dot(w, X[i]) <= 0: 
                w = w + r * y[i] * X[i]
                
    return w


# In[98]:


w = perceptron(train, 0.1, 10)
print("w = ", w)

n_error = 0

for i in range(len(test)):
    prediction = np.sign(w.dot(test[i][:-1]))
    if prediction != 2 * test[i][-1] - 1:
        n_error += 1

print("Number of misclassified tests =", n_error)
print("error =", n_error/len(test))


# # Voted perceptron
# 

# In[83]:


train = []
with open ("/Users/sinarashetnia/Desktop/bank-note/train.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a = [1] + list(map(float,a))
        train.append(a)

train = np.array(train)


# In[84]:


test = []
with open ("/Users/sinarashetnia/Desktop/bank-note/test.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a =[1] + list(map(float,a))
        test.append(a)

test = np.array(test)


# In[85]:


def voted_perceptron(D , r, T):
    w = np.zeros(len(D[0] )- 1)
    m = 0
    w_list = [w]
    c = []
    for t in range(T):
        np.random.shuffle(D)
        X = D[:,:-1]
        y = 2* D[:,-1] -1
        for i in range(len(D)):
            if y[i] * w.dot(X[i]) <= 0:
                w = w + r*y[i]*X[i]
                m = m+1
                c.append(1)
                w_list.append(w)
            else:
                c[-1] += 1 
            
    return c, w_list[1:]  


# In[86]:


c, w_list = voted_perceptron(train , 0.1, 20)


# In[87]:


len(w_list), len(c)


# In[88]:


def prediction(w, c, x):
    a = sum([c[i] * np.sign(w[i].dot(x)) for i in range(len(c))])
    return np.sign(a)
    


# In[89]:


c , w = voted_perceptron(train , 0.1, 20)
n_error= 0
for i in range(len(test)):
    if prediction(w,c,test[i][:-1]) != 2*test[i][-1] -1:
        n_error += 1
print(n_error)
print(n_error/len(test))
    


# # Average perceptron

# In[90]:


train = []
with open ("/Users/sinarashetnia/Desktop/bank-note/train.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a = [1] + list(map(float,a))
        train.append(a)

train = np.array(train)


# In[91]:


test = []
with open ("/Users/sinarashetnia/Desktop/bank-note/test.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a =[1] + list(map(float,a))
        test.append(a)

test = np.array(test)


# In[92]:


def average_perception(D , r, T):
    w = np.zeros(len(D[0])- 1)
    a = np.zeros(len(D[0])-1)
    for t in range(T):
        np.random.shuffle(D)
        X = D[:,:-1]
        y = 2* D[:,-1] -1
        for i in range(len(D)):
            if y[i] * w.dot(X[i]) <= 0:
                w = w + r * y[i] * X[i]
            
            a = a + w
            
    return a


# In[93]:


def prediction(a , x):
    return np.sign(a.dot(x))


# In[94]:


a = average_perception(train , 0.1, 20)
n_error = 0
for i in range(len(test)):
    if prediction(a,test[i][:-1]) != 2*test[i][-1] -1:
        n_error += 1
print(n_error)
print(n_error/len(test))


# In[ ]:




