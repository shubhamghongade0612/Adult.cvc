#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('dark_background')


# In[2]:


df=pd.read_csv("adult.csv")


# In[3]:


df.head()


# In[4]:


header_names=['Age','Workclass','Fnlwgt','Education','education_num','marital_status','occupation','relationship','race'
              ,'sex','capital_gain','capital_loss','hours_per_week','native_country','income']
df=pd.read_csv("adult.csv",header=None,skiprows=1,names=header_names)


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


print(df.columns)


# In[10]:


df['income'].value_counts()


# In[11]:


df['sex'].value_counts()


# In[12]:


df['native_country'].value_counts()


# In[13]:


df['Workclass'].value_counts()


# In[14]:


df['occupation'].value_counts()


# In[15]:


df=df.drop(['Education','Fnlwgt'],axis=1)
df.head()


# In[16]:


df.replace('?',np.NaN,inplace = True)
df.head()


# In[17]:


df.fillna(method='ffill',inplace=True)


# In[18]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Workclass']=le.fit_transform(df['Workclass'])
df['marital_status']=le.fit_transform(df['marital_status'])
df['occupation']=le.fit_transform(df['occupation'])
df['relationship']=le.fit_transform(df['relationship'])
df['race']=le.fit_transform(df['race'])
df['sex']=le.fit_transform(df['sex'])
df['native_country']=le.fit_transform(df['native_country'])
df['income']=le.fit_transform(df['income'])
df.head()


# In[19]:


sns.barplot(x='income',y='Age',data=df)
#so here in 0 means less than 50k and 1=>50k that means people with less age are earning less than 50k and 
#more aged person are earning more


# In[20]:


sns.pairplot(df,hue='income',palette='inferno')


# In[21]:


sns.heatmap(df.corr())


# In[24]:


x=df.drop(['income'],axis=1)
y=df['income']


# Train Test Split

# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[29]:


from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(x_train,y_train)


# In[30]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# # Printing Metrics

# In[31]:


y_pred=gb.predict(x_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)*100)


# so the accurracy of the data is 79.6714% as if on current scale by arranging the data and after removing some conclusions....

# Thank You
