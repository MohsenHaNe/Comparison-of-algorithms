#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris as dataset
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 


# In[2]:


data = dataset()
x = data.data
y = data.target


# In[3]:


x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size=0.3)


# In[4]:


KNN = KNeighborsClassifier()


# In[5]:


KNN.fit(x_train , y_train )


# In[6]:


random_forest = RandomForestClassifier()


# In[7]:


random_forest.fit(x_train , y_train )


# In[8]:


svm = SVC()


# In[9]:


svm.fit(x_train , y_train )


# In[10]:


knn_pred = KNN.predict(x_test)


# In[11]:


random_forest_pred = random_forest.predict(x_test)


# In[12]:


svm_pred = svm.predict(x_test)


# In[13]:


knn_acc = accuracy_score(y_test, knn_pred)
random_forest_acc = accuracy_score(y_test, random_forest_pred)
svm_acc = accuracy_score(y_test, svm_pred)


# In[14]:


np.round(knn_acc , 2)


# In[15]:


np.round(random_forest_acc , 2)


# In[16]:


np.round(svm_acc , 2)


# In[19]:


knn_score= cross_val_score(KNN, x, y, cv=5)
random_forest_score = cross_val_score(random_forest, x, y, cv=5)
svm_score = cross_val_score(svm, x, y, cv=5)


# In[20]:


knn_score


# In[21]:


random_forest_score


# In[22]:


svm_score


# In[ ]:




