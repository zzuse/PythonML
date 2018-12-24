#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.datasets import load_iris


# In[23]:


iris_dataset = load_iris()


# In[14]:


print('keys of iris_datasets: \n{}'.format(iris_dataset.keys()))


# In[15]:


print(iris_dataset['DESCR'][:193] + "\n...")


# In[16]:


print(iris_dataset['target_names'])


# In[6]:


print(iris_dataset['feature_names'])


# In[7]:


print(iris_dataset['data'].shape)


# In[9]:


print(iris_dataset['target'])


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)


# In[18]:


print(X_train.shape)


# In[19]:


print(y_train.shape)


# In[20]:


print(X_test.shape)


# In[21]:


print(y_test.shape)


# In[10]:


import pandas as pd


# In[28]:


import pip


# In[33]:


import mglearn


# In[52]:


import numpy as np


# In[35]:


iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)


# In[45]:


grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[47]:


from sklearn.neighbors import KNeighborsClassifier


# In[48]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[56]:


knn.fit(X_train, y_train)


# In[50]:


KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')


# In[53]:


X_new=np.array([[5,2.9,1,0.2]])


# In[54]:


print("X_new.shape:{}".format(X_new.shape))


# In[57]:


prediction = knn.predict(X_new)


# In[58]:


print("Prediction: {}".format(prediction))
print("predicted target name: {}".format(iris_dataset['target_names'][prediction]))


# In[59]:


y_pred = knn.predict(X_test)


# In[60]:


print("Test set prediction:\n {}".format(y_pred))


# In[61]:


print("Test set score: {:.2f}".format(np.mean(y_pred==y_test)))


# In[62]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:




