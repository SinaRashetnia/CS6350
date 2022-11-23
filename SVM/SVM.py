#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd 
import random 
import scipy.optimize
from scipy.optimize import minimize


# In[7]:


train = []
with open ("/Users/sinarashetnia/Desktop/bank-note/train.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a = [1] + list(map(float,a))
        train.append(a)

train = np.array(train)
train.shape


# In[8]:


test = []
with open ("/Users/sinarashetnia/Desktop/bank-note/test.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a =[1] + list(map(float,a))
        test.append(a)

test = np.array(test)
test.shape


# In[9]:


def gamma(t, gamma_0, a):
    k = gamma_0 / (1 + (gamma_0 / a) *t)
    return k


# In[10]:


def Stochastic_gradient_descent_SVM(s, T, c, gamma_0 , a):
    w = np.zeros(len(s[0]) - 1)  # e.g. 6 - 1 = 5
    n = len(s) # 872
    for t in range(T):
        np.random.shuffle(s)
        X = s[:, :-1] # train
        y = 2 * s[:, -1] - 1 # labels
        for i in range(n):
            if y[i] * np.dot(w, X[i]) <= 1:
                w = w - gamma(t, gamma_0, a) * w + gamma(t, gamma_0, a) * c * n * y[i] * X[i]
            else:
                w[:-1] = (1 - gamma(t, gamma_0, a)) * w[:-1]
    return w


# In[11]:


c_0 = [100/872 , 500/872, 700/872]
a_0 = [1]
t = 100
lr = []
gamma_1 = [1]
for c in c_0:
    for gamma_0 in gamma_1:
        for a in a_0:
            w = Stochastic_gradient_descent_SVM(train, 100, gamma_0 , a , c)
            print("w = ", w)
            print()
            n_error = 0

            for i in range(len(test)):
                prediction = np.sign(w.dot(test[i][:-1]))
                if prediction != 2 * test[i][-1] - 1:
                    n_error += 1

            print("Number of misclassified tests =", n_error)
            print("error =", n_error/len(train))
            print()
            for i in range(len(train)):
                prediction = np.sign(w.dot(train[i][:-1]))
                if prediction != 2 * train[i][-1] - 1:
                    n_error += 1

            print("Number of misclassified train =", n_error)
            print("error =", n_error/len(train))


# In[16]:


c_0 = [100/872 , 500/872, 700/872]
t = 100
lr = []
gamma_1 = [1]
for c in c_0:
    for gamma_0 in gamma_1:
        w = Stochastic_gradient_descent_SVM(train, 100, gamma_0 , gamma_0 , c)
        print("w = ", w)
        print()
        n_error = 0

        for i in range(len(test)):
            prediction = np.sign(w.dot(test[i][:-1]))
            if prediction != 2 * test[i][-1] - 1:
                n_error += 1

        print("Number of misclassified tests =", n_error)
        print("error =", n_error/len(train))
        print()
        for i in range(len(train)):
            prediction = np.sign(w.dot(train[i][:-1]))
            if prediction != 2 * train[i][-1] - 1:
                n_error += 1

        print("Number of misclassified train =", n_error)
        print("error =", n_error/len(train))


# # Dual SVM

# In[1]:


import numpy as np
import pandas as pd 
import random 
import scipy.optimize
from scipy.optimize import minimize


# In[2]:


train = []
with open ("/Users/sinarashetnia/Desktop/bank-note/train.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a = [1] + list(map(float,a))
        train.append(a)

train = np.array(train)
train.shape


# In[3]:


test = []
with open ("/Users/sinarashetnia/Desktop/bank-note/test.csv", "r") as file:
    for line in file:
        a = line.strip().split(",")
        a =[1] + list(map(float,a))
        test.append(a)

test = np.array(test)
test.shape


# In[4]:


Y = 2 * train[:, -1] -1
X = train[:, :-1]


# In[5]:


def SVM_dual_function(alpha):
    Y0 = np.diag(Y)
    return (alpha @ ((Y0 @ train[:, :-1])@(Y0 @ train[:, :-1]).T) @ alpha.T - sum(alpha))/2


# In[6]:


n = len(train)
m = len(test)

def SVM_dual(C):
    ans = minimize(SVM_dual_function, 
                   np.zeros(n), 
                   method='SLSQP', 
                   bounds=tuple([(0,C) for i in range(n)]), 
                   constraints={'type':'eq', 
                                'fun': lambda alpha: np.dot(alpha, train[:, -1])}).x
    return ans


# In[7]:


min_of_a = []
C_0 = [100/873, 500/873, 700/873]

for C in C_0:
    Min = SVM_dual(C)
    min_of_a.append(Min)


# In[8]:


W = []
for i in range(len(C_0)):
    W.append(sum(min_of_a[i][j] * Y[j] * X[j] for j in range(n)))


# In[9]:


bias_1 = []
bias_2 = []
bias_3 = []
B = [bias_1, bias_2, bias_3]

for i in range(len(C_0)):
    for j in range(n):
        if 1e-6 < min_of_a[i][j] < C_0[i] - 1e-6: 
            B[i].append(Y[j]- W[i].dot(X[j]))

B


# In[12]:


average_of_b = [np.mean(B[i]) for i in range(3)]
print(average_of_b)


# In[13]:


def sgn(x):
    if x >=0:
        return 1
    else:
        return -1


# In[14]:


def predict(x, C):
    for j in range(len(C_0)):
        if C == C_0[j]:
            l = j
    return sgn((W[l].T @ x) + average_of_b[l])


# In[15]:


final_train = np.ones((len(C_0), n))

for j in range(len(C_0)):
    for i in range(n):
        final_train[j][i] = predict(X[i], C_0[j])


# In[16]:


c = np.zeros(len(C_0))

for j in range(len(C_0)):
    for i in range(n):
        if final_train[j][i] != Y[i]:
            c[j] = c[j] + 1
print("Train error =", c/len(X))
print("Number of missclassified train examples:", list(c))


# In[17]:


final_test = np.ones((len(C_0), m))

for j in range(len(C_0)):
    for i in range(m):
        final_test[j][i] = predict(test[i][:-1], C_0[j])


# In[18]:


f = np.zeros(len(C_0))

for j in range(len(C_0)):
    for i in range(m):
        if final_test[j][i] != test[i][-1]:
            f[j] = f[j] + 1
print("Train error =", f/len(test))
print("Number of missclassified test examples:", list(f))


# In[ ]:




