#!/usr/bin/env python
# coding: utf-8

# # Title: Breast Cancer Tumer Classification

# ## Import Important Libraries

# In[1]:


from warnings import filterwarnings
filterwarnings('ignore')

import os
os.chdir("C:/Users/Aniket/OneDrive/Desktop/A project/")

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,roc_auc_score,mean_squared_error,r2_score,mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sn


# # Data Collection

# In[2]:


df=pd.read_csv('cancer data.csv')
df.head()


# # EDA

# In[3]:


df.info()


# In[4]:


df.isna().sum()


# # Define X and Y

# In[5]:


x=df.drop(['id','diagnosis'],axis=1)
y=df['diagnosis']


# In[6]:


x


# In[7]:


y


# In[8]:


# Create a dictionary to map 'M' to 1 and 'B' to 0
mapping = {'M': 1, 'B': 0}

# Use the map function to apply the mapping to your variable
y1 = [mapping[value] for value in y]


# In[9]:


y1


# # Spliting dataset

# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x,y1,test_size=0.2,random_state=21)


# # Algorithm Evaluation

# In[11]:


lr=LogisticRegression()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
ab=AdaBoostClassifier()
knn=KNeighborsClassifier()
svm=SVC()


# In[12]:


List1=[lr,dt,rf,ab,knn,svm]


# In[13]:


for i in List1:
    i.fit(x_train,y_train)
    
    y_pred_train=i.predict(x_train)
    y_pred=i.predict(x_test)
    
    tr=f1_score(y_pred_train,y_train)
    ts=f1_score(y_pred,y_test)
    
    print('*'*50)
    print(i)
    print('Training F1 score: ',tr)
    print('Testing F1 score: ',ts)
    


# # Feature Selection

# In[14]:


rfs=SequentialFeatureSelector(rf,direction='forward')


# In[15]:


rfs.fit_transform(x,y1)


# In[17]:


cols=rfs.get_feature_names_out()
cols


# In[18]:


x2=pd.DataFrame(x,columns=cols)
x2


# # Spliting dataset

# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x2,y1,test_size=0.2,random_state=21)


# # Model Building using new feratues

# In[22]:


model=rf.fit(x_train,y_train)
    
y_pred_train=model.predict(x_train)
y_pred=model.predict(x_test)
    
tr=f1_score(y_pred_train,y_train)
ts=f1_score(y_pred,y_test)

print('Training F1 score: ',tr)
print('Testing F1 score: ',ts)


# In[ ]:




