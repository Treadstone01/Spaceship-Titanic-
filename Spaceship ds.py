#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('spaceship_train.csv')


# In[3]:


df.info()


# In[4]:


df.head(10)


# In[5]:


# checking sparsity of null values 
plt.figure(figsize = (10,6))
sns.heatmap(df.isna(), yticklabels = False, cbar = False )


# In[6]:


# checking the correlation of columns 
plt.figure(figsize= (10,6))
sns.heatmap(df.corr(), annot = True)


#  name - not useful in anyway 
# 
# passenger_id - some sort of index, appears irrelevant at first glance but there's some correlation between the groups and the target variable
# 
# cabin seems to be useless at first glance but with feature engineering , might be able to extract useful data 
# 

# In[7]:


df.isna().value_counts()


# of 8692 data entries, 6606 have no null values for all features 
# 
# it is safe to drop null rows since we cannot pick random values for cryosleep,destination and vip because they affect the target 

# In[8]:


df.dropna(inplace = True)


# In[9]:


df.info()


# In[10]:


df.isnull().value_counts()


# In[11]:


# to drop the irrelevant columns 
df.drop('Name', axis = 1 ,inplace =  True)


# In[12]:


# extracting the group numbers from the passengerid 
df['Passenger'] = [i.split('_') for i in df['PassengerId']]


# In[13]:


df['Passenger'] = [j[0] for j in df['Passenger']]
df['Passenger']


# In[14]:


# feature extraction for Cabin column , the deck and and side 
df['Cabins'] = [c.split('/') for c in df['Cabin']]
df['Cabins'] = [d[0] + d[-1] for d in df['Cabin']]
df['Cabins']


# In[15]:


df['Cabins'].value_counts()


# In[16]:


df.drop('Cabin', axis = 1 , inplace = True)


# In[17]:


df = df.reset_index(drop = True)


# In[18]:


df.drop('PassengerId', axis = 1 , inplace = True)


# In[19]:


df.head()


# In[20]:


df['CryoSleep'] = pd.get_dummies(df['CryoSleep'], drop_first = True)


# In[21]:


df['VIP'] = pd.get_dummies(df['VIP'], drop_first = True)


# In[22]:


df


# In[23]:


df['HomePlanet'].value_counts()


# In[24]:


df[['Home_europa', 'Home_mars']] = pd.get_dummies(df['HomePlanet'], drop_first = True)
df


# In[25]:


df.drop('HomePlanet' , axis = 1 , inplace = True)


# In[26]:


df.info()


# In[27]:


df['Destination'].value_counts()


# In[28]:


df[['Destination_55 Cancri e', 'Destination_P50 J318.5-22']] = pd.get_dummies(df['Destination'], drop_first = True)


# In[29]:


df.drop('Destination', axis = 1, inplace = True)


# In[30]:


df.head()


# In[31]:


# checking the bottom of the bucket_list variables of df['Cabin']
for c in df['Cabins']:
    if c == 'TP': 
        print(df.loc[df['Cabins'] == 'TP'])


# both entries were going from europa to Trappist- 1e , a very close age range , both didn't take the cryosleep program , one seemed to be an extravagant spend and was not transported...lol

# In[32]:


# to manually encode the Cabins column cause it has so many variables and we don't want hierarchy with label encoding , we'll be creating a dictionary
dict1 = {'FP' : 1 ,'FS'  : 2 , 'GP'  : 3 ,'GS'  : 4 ,'ES'  : 5 , 'BS'  : 6 ,  'EP'  : 7 ,'BP'  : 8 ,'CP'  : 9 ,'DP'  : 10,'DS'  : 11,'AS'  : 12, 'AP'  : 13,  'TP'  : 14 }


# In[33]:


df['Cabin'] = df.Cabins.map(dict1)


# In[34]:


df['Cabin'].value_counts()


# In[35]:


df.drop('Cabins', axis = 1 , inplace = True)


# In[36]:


df.head()


# In[37]:


sns.set_theme()
plt.figure(figsize = (16,8))
sns.countplot(x = 'Transported', hue = "Cabin", data = df)


# In[38]:


df.head()


# In[39]:


plt.figure(figsize = (10,6))
sns.heatmap(df.corr(), annot = True)


# In[40]:


sns.heatmap(df.isnull(), cbar = False, yticklabels = False)


# In[41]:


df.dropna(inplace = True)
sns.heatmap(df.isnull(), cbar = False, yticklabels = False)


# In[42]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split 


# In[43]:


X = df.drop('Transported', axis = 1 )
y = df['Transported']


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2)


# In[45]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random_state = 2)
param_grid1 = [
    { 'n_estimators' : [8,10,15,20],
     'max_features' : ['auto' , 'sqrt' , 'log2'],
     'max_depth' : [6],
     'min_samples_split' : [2,5],
     'min_samples_leaf': [1,4],
     'min_weight_fraction_leaf' : np.log(range(1,5))
    }
]
cff = GridSearchCV(rf, param_grid= param_grid1 , cv = 5 , verbose = 2 ,n_jobs = 4 )
cff.fit(X_train , y_train)


# In[48]:


dfx = cff.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(dfx, y_test), '\n',confusion_matrix(dfx,y_test) )


# In[49]:


dff = pd.DataFrame(cff.cv_results_)


# In[50]:


print(cff.best_params_)
print('\n')
cff.best_score_


# In[51]:


rfc = RandomForestClassifier(random_state = 2, n_estimators = 32 )
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(pred, y_test), '\n',confusion_matrix(pred,y_test) )


# In[52]:


from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(random_state = 42 )


# In[53]:


param_grid1 = [
    {  'criterion' : ['gini', 'entropy'],
       'splitter' : ['best', 'random'],
       'min_samples_split' : [2, 10 , 15,20],
       'min_samples_leaf' : [1,3,6,7],
    }
]
clf = GridSearchCV(dc, param_grid= param_grid1 , cv = 5 , verbose = 2 ,n_jobs = 4 )
ddc = clf.fit(X_train , y_train)
print(clf.best_params_)
print('\n')
print(clf.best_score_)
ddc = clf.predict(X_test)
print(classification_report(ddc, y_test), '\n',confusion_matrix(ddc,y_test) )


# In[54]:


dc = DecisionTreeClassifier(random_state = 42, criterion = 'gini' , min_samples_leaf = 7 , min_samples_split = 15 , splitter = 'random' )
dc.fit(X_train, y_train)
ddc = dc.predict(X_test)
print(classification_report(ddc, y_test), '\n',confusion_matrix(ddc,y_test) )


# In[55]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state = 42)
param_grid  = [
    {'penalty':['l1', 'l2' , 'elasticnet' , 'none'],
     'C' : np.logspace(-1, -5, 1 ),
     'solver': ['lbfgs' , 'newton-cg', 'liblinear' , 'sag' , 'saga'],
     'max_iter' : [10, 20 , 30]
    }
]


# In[56]:


lrg = GridSearchCV(lg, param_grid= param_grid , cv = 5 , verbose = 2 ,n_jobs = 4 )
lrg.fit(X_train , y_train)
print(lrg.best_params_)
print('\n')
print(lrg.best_score_)
h = lrg.predict(X_test)
print(classification_report(h, y_test), '\n',confusion_matrix(h,y_test) )


# In[57]:


lrg.best_estimator_


# In[58]:


lg = pd.DataFrame(lrg.cv_results_)
lg[['param_C','param_max_iter', 'param_penalty','param_solver', 'mean_test_score', 'rank_test_score']]


# In[59]:


lrx = LogisticRegression(random_state = 42, C = 0.2 , max_iter = 10 , solver = 'newton-cg' , penalty = 'none' )
lrx.fit(X_train,y_train)
predl = lrx.predict(X_test)
print(classification_report(predl, y_test), '\n',confusion_matrix(predl,y_test) )

