#!/usr/bin/env python
# coding: utf-8

# In[157]:


import numpy as np
from numpy import genfromtxt
from scipy.special import expit as logistic
from scipy.optimize import minimize
from scipy.stats import norm


# In[158]:


reg = 1


# In[159]:


data = genfromtxt("/Users/sinarashetnia/Desktop/bank-note/train.csv", delimiter=',')
test = genfromtxt("/Users/sinarashetnia/Desktop/bank-note/test.csv", delimiter=',')


# In[160]:


data.shape, test.shape


# In[161]:


train_data = data[:,:-1]
train_data = np.hstack((np.ones((train_data.shape[0],1)), train_data))
train_label = data[:,-1].astype(int)

test_data = test[:,:-1]
test_data = np.hstack((np.ones((test_data.shape[0],1)), test_data))
test_label = test[:,-1].astype(int)


# In[162]:


train_data.shape, train_label.shape


# ### Logistic Regression 
# 

# In[163]:


sigmoid = lambda x: 1/(1+np.exp(-x))


# In[164]:


def grad_logistic(x, y, w, reg):
    if y == 1:
        grad = sigmoid(-np.dot(w,x))*x
    else:
        grad = -sigmoid(np.dot(w,x))*x
    return reg * w -872*grad


# In[165]:


def gamma(t, gamma_0, d):
    k = gamma_0/(1 + t*gamma_0/d)
    return k


# In[166]:


def Stochastic_gradient_descent(X, y, gamma_0 , d, epoch, reg):
    n, m = X.shape
    w = np.zeros(m)  # e.g. 6 - 1 = 5

    List = list(range(n))
    for t in range(epoch):
        np.random.shuffle(List)
        X_1 = X[List,:]
        y_1 = y[List]
        for j in range(n):
            w = w - gamma(t, gamma_0, d) * grad_logistic(X_1[j], y_1[j], w, reg)
    return w


# In[167]:


w = Stochastic_gradient_descent(train_data, train_label, gamma_0 = .1 , d=.1, epoch= 100, reg = 1)


# In[168]:


def pred(X, w):
    score = X @ w
    return logistic(score)


# In[169]:


y_hat = (pred(train_data, w)>0.5)
(y_hat == train_label).mean()


# In[170]:


y_hat = (pred(test_data, w)>0.5)
(y_hat == test_label).mean()


# In[171]:


for varience in {0.01 , 0.1 , 0.5, 1, 3, 5, 10, 100}:
    w = Stochastic_gradient_descent(train_data, train_label, gamma_0 = 0.01 , d=0.001, epoch= 100, reg = 1/varience)
    
    y_hat = (pred(train_data, w)>0.5)
    e_tr = 1 -(y_hat == train_label).mean()
    
    y_hat = (pred(test_data, w)>0.5)
    e_te = 1-(y_hat == test_label).mean()
    print(e_tr,e_te )


# In[174]:


for varience in {0.01 , 0.1 , 0.5, 1, 3, 5, 10, 100}:
    w = Stochastic_gradient_descent(train_data, train_label, gamma_0 = 0.001 , d=0.002, epoch= 100, reg = 0)
    y_hat = (pred(train_data, w)>0.5)
    e_tr = 1-(y_hat == train_label).mean()
    y_hat = (pred(test_data, w)>0.5)
    e_te = 1-(y_hat == test_label).mean()
    print(e_tr,e_te )


# ### PyTorch

# In[ ]:

#conda install pytorch torchvision -c pytorch


# In[103]:


import torch
import torch.nn as nn
import torch.optim as opt


# In[104]:


data = genfromtxt("/Users/sinarashetnia/Desktop/bank-note/train.csv", delimiter=',')
test = genfromtxt("/Users/sinarashetnia/Desktop/bank-note/test.csv", delimiter=',')
train_data = data[:,:-1]
train_label = data[:,-1].astype(int)

test_data = test[:,:-1]
test_label = test[:,-1].astype(int)


# In[127]:


X = torch.from_numpy(train_data).type(torch.float32)
y = torch.from_numpy(train_label).type(torch.int64)
X_1 = torch.from_numpy(test_data).type(torch.float32)
y_1 = torch.from_numpy(test_label).type(torch.int64)


# In[128]:


X.shape


# In[129]:


class NeuralNet_depth_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)  
        self.act = activation
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        return out


# In[130]:


class NeuralNet_depth_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)  
        self.act = activation
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, x_1):
        out = self.fc1(x_1)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        return out


# In[131]:


num_epochs = 100


# In[132]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_3(input_size = 4, hidden_size = h, num_classes = 2, activation = torch.tanh)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.1)
    
    for epoch in range(num_epochs): 
        y_h = model(X)
        loss = criterion(y_h, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X) 
    e = 1 - (y_hat.max(axis = 1)[1] == y).sum()/X.shape[0]
    print('error for depth = 3 and width = {}, train error = {}'.format(h, e))


# In[175]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_3(input_size = 4, hidden_size = h, num_classes = 2, activation = torch.tanh)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.1)
    
    for epoch in range(num_epochs): 
        y_h = model(X_1)
        loss = criterion(y_h, y_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X_1) 
    e = 1 - (y_hat.max(axis = 1)[1] == y_1).sum()/X_1.shape[0]
    print('error for depth = 3 and width = {}, test error = {}'.format(h, e))


# In[134]:


class NeuralNet_depth_5(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_5, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        self.act = activation
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.act(out)
        
        out = self.fc5(out)
    
        return out


# In[136]:


class NeuralNet_depth_5(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_5, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        self.act = activation
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)
    
    def forward(self, x_1):
        out = self.fc1(x_1)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.act(out)
        
        out = self.fc5(out)
    
        return out


# In[137]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_5(input_size = 4, hidden_size = h, num_classes = 2, activation = torch.tanh)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.1)
    
    for epoch in range(num_epochs): 
        y_h = model(X)
        loss = criterion(y_h, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X) 
    e = 1 - (y_hat.max(axis = 1)[1] == y).sum()/X.shape[0]
    print('error for depth = 5 and width = {}, train error = {}'.format(h, e))


# In[139]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_5(input_size = 4, hidden_size = h, num_classes = 2, activation = torch.tanh)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.01)
    
    for epoch in range(num_epochs): 
        y_h = model(X_1)
        loss = criterion(y_h, y_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X_1) 
    e = 1 - (y_hat.max(axis = 1)[1] == y_1).sum()/X_1.shape[0]
    print('error for depth = 5 and width = {}, test error = {}'.format(h, e))


# In[112]:


class NeuralNet_depth_9(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_9, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, num_classes)
        self.act = activation
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.xavier_normal_(self.fc6.weight)
        nn.init.xavier_normal_(self.fc7.weight)
        nn.init.xavier_normal_(self.fc8.weight)
        nn.init.xavier_normal_(self.fc9.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.act(out)
        
        out = self.fc5(out)
        out = self.act(out)
        
        out = self.fc6(out)
        out = self.act(out)
        
        out = self.fc7(out)
        out = self.act(out)
        
        out = self.fc8(out)
        out = self.act(out)
        
        
        out = self.fc9(out)
    
        return out


# In[ ]:


class NeuralNet_depth_9(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_9, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, num_classes)
        self.act = activation
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.xavier_normal_(self.fc6.weight)
        nn.init.xavier_normal_(self.fc7.weight)
        nn.init.xavier_normal_(self.fc8.weight)
        nn.init.xavier_normal_(self.fc9.weight)
    
    def forward(self, x_1):
        out = self.fc1(x_1)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.act(out)
        
        out = self.fc5(out)
        out = self.act(out)
        
        out = self.fc6(out)
        out = self.act(out)
        
        out = self.fc7(out)
        out = self.act(out)
        
        out = self.fc8(out)
        out = self.act(out)
        
        
        out = self.fc9(out)
    
        return out


# In[114]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_9(input_size = 4, hidden_size = h, num_classes = 2, activation = torch.tanh)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.01)
    
    for epoch in range(num_epochs): 
        y_h = model(X)
        loss = criterion(y_h, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X) 
    e = 1 - (y_hat.max(axis = 1)[1] == y).sum()/X.shape[0]
    print('error for depth = 9 and width = {}, train error = {}'.format(h, e))


# In[140]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_9(input_size = 4, hidden_size = h, num_classes = 2, activation = torch.tanh)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.01)
    
    for epoch in range(num_epochs): 
        y_h = model(X_1)
        loss = criterion(y_h, y_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X_1) 
    e = 1 - (y_hat.max(axis = 1)[1] == y_1).sum()/X_1.shape[0]
    print('error for depth = 9 and width = {}, test error = {}'.format(h, e))


# In[85]:


class NeuralNet_depth_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)  
        self.act = activation
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        return out


# In[141]:


class NeuralNet_depth_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)  
        self.act = activation
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
    
    def forward(self, x_1):
        out = self.fc1(x_1)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        return out


# In[116]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_3(input_size = 4, hidden_size = h, num_classes = 2, activation = nn.ReLU())
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.1)
    
    for epoch in range(num_epochs): 
        y_h = model(X)
        loss = criterion(y_h, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X) 
    e = 1 - (y_hat.max(axis = 1)[1] == y).sum()/X.shape[0]
    print('error for depth = 3 and width = {}, train error = {}'.format(h, e))


# In[176]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_3(input_size = 4, hidden_size = h, num_classes = 2, activation = nn.ReLU())
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.1)
    
    for epoch in range(num_epochs): 
        y_h = model(X_1)
        loss = criterion(y_h, y_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X_1) 
    e = 1 - (y_hat.max(axis = 1)[1] == y_1).sum()/X_1.shape[0]
    print('error for depth = 3 and width = {}, test error = {}'.format(h, e))


# In[143]:


class NeuralNet_depth_5(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_5, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        self.act = activation
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.act(out)
        
        out = self.fc5(out)
    
        return out


# In[144]:


class NeuralNet_depth_5(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_5, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        self.act = activation
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
    
    def forward(self, x_1):
        out = self.fc1(x_1)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.act(out)
        
        out = self.fc5(out)
    
        return out


# In[145]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_5(input_size = 4, hidden_size = h, num_classes = 2, activation = nn.ReLU())
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.1)
    
    for epoch in range(num_epochs): 
        y_h = model(X)
        loss = criterion(y_h, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X) 
    e = 1 - (y_hat.max(axis = 1)[1] == y).sum()/X.shape[0]
    print('error for depth = 5 and width = {}, train error = {}'.format(h, e))


# In[147]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_5(input_size = 4, hidden_size = h, num_classes = 2, activation = nn.ReLU())
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.01)
    
    for epoch in range(num_epochs): 
        y_h = model(X_1)
        loss = criterion(y_h, y_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X_1) 
    e = 1 - (y_hat.max(axis = 1)[1] == y_1).sum()/X_1.shape[0]
    print('error for depth = 5 and width = {}, test error = {}'.format(h, e))


# In[ ]:


class NeuralNet_depth_9(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_9, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, num_classes)
        self.act = activation
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.kaiming_normal_(self.fc6.weight)
        nn.init.kaiming_normal_(self.fc7.weight)
        nn.init.kaiming_normal_(self.fc8.weight)
        nn.init.kaiming_normal_(self.fc9.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.act(out)
        
        out = self.fc5(out)
        out = self.act(out)
        
        out = self.fc6(out)
        out = self.act(out)
        
        out = self.fc7(out)
        out = self.act(out)
        
        out = self.fc8(out)
        out = self.act(out)
        
        
        out = self.fc9(out)
    
        return out


# In[148]:


class NeuralNet_depth_9(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(NeuralNet_depth_9, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, num_classes)
        self.act = activation
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.kaiming_normal_(self.fc6.weight)
        nn.init.kaiming_normal_(self.fc7.weight)
        nn.init.kaiming_normal_(self.fc8.weight)
        nn.init.kaiming_normal_(self.fc9.weight)
    
    def forward(self, x_1):
        out = self.fc1(x_1)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.act(out)
        
        out = self.fc5(out)
        out = self.act(out)
        
        out = self.fc6(out)
        out = self.act(out)
        
        out = self.fc7(out)
        out = self.act(out)
        
        out = self.fc8(out)
        out = self.act(out)
        
        
        out = self.fc9(out)
    
        return out


# In[151]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_9(input_size = 4, hidden_size = h, num_classes = 2, activation = nn.ReLU())
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.02)
    
    for epoch in range(num_epochs): 
        y_h = model(X)
        loss = criterion(y_h, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X) 
    e = 1 - (y_hat.max(axis = 1)[1] == y).sum()/X.shape[0]
    print('error for depth = 9 and width = {}, train error = {}'.format(h, e))


# In[180]:


for h in [5 , 10 ,25, 50, 100]:
    model = NeuralNet_depth_9(input_size = 4, hidden_size = h, num_classes = 2, activation = nn.ReLU())
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.002)
    
    for epoch in range(num_epochs): 
        y_h = model(X_1)
        loss = criterion(y_h, y_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X_1) 
    e = 1 - (y_hat.max(axis = 1)[1] == y_1).sum()/X_1.shape[0]
    print('error for depth = 9 and width = {}, test error = {}'.format(h, e))


# Problem 1

# In[153]:





# In[154]:





# In[156]:





# In[ ]:




