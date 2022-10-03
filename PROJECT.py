#!/usr/bin/env python
# coding: utf-8

# In[57]:


#Importating the dependencies
import numpy as np 
import pandas as pd


# In[41]:


df=pd.read_csv(r"C:\Users\MONIKA\Desktop\Levels_Fyi_Salary_Data.csv")
df


# In[42]:


df.columns


# In[43]:


import numpy as np 
import pandas as pd
salaryDF=df[['totalyearlycompensation','yearsofexperience','basesalary','yearsatcompany']]
print(salaryDF.head())


# # Data preprocessing

# In[44]:


salaryDF.shape


# In[45]:


salaryDF.describe


# In[46]:


salaryDF.info


# In[47]:


salaryDF.isnull().sum()


# In[48]:


df.duplicated().sum()


# # Data vizualization

# In[9]:


salaryDF.corr()


# In[10]:


x=salaryDF.drop('totalyearlycompensation',axis=1)
y=salaryDF['totalyearlycompensation']
x


# In[11]:


y


# In[12]:


x.shape


# # importing necessary libraries

# In[13]:



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# # splitting the data into training data and testing data

# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[15]:


x_test


# In[16]:


x_train


# In[17]:


y_train


# In[18]:


y_test=np.array(y_test)
y_test


# # Model evaluation using LinearRegression algorithm

# In[19]:


model = LinearRegression()


# In[20]:


model.fit(x_test, y_test)
r_train=model.score(x_train,y_train)
r_test=model.score(x_test,y_test)


# In[21]:


r_train


# In[22]:


r_test


# In[23]:


y_pred=model.predict(x_test)
y_pred


# In[24]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


# # Building predictive system

# In[51]:


y_pred=model.predict(np.array([6,173000,0]).reshape(1,-1))
y_pred[0]


# # Model evaluation using DecisionTreeClassifier

# In[26]:


regressor= DecisionTreeClassifier()


# In[27]:


regressor.fit(x_test, y_test)
m_train=regressor.score(x_train,y_train)
m_test=regressor.score(x_test,y_test)


# In[28]:


x_train


# In[29]:


y_train


# In[30]:


x_test


# In[31]:


y_test


# In[32]:


m_test


# In[33]:


y_pred=regressor.predict(x_test)
y_pred


# In[34]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


# # Building predictive system

# In[56]:


y_pred=regressor.predict(np.array([1.5,107000,1.5]).reshape(1,-1))
y_pred[0]


# In[ ]:




