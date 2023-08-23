#!/usr/bin/env python
# coding: utf-8

# # ** Importing Packages and Data**

# In[29]:


# Ignore warnings
import warnings
warnings.simplefilter("ignore")

# Frequently using packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Useful data handling package
get_ipython().system(' pip install dfply -q')
from dfply import *

# Data Analyses
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # **2-2. Exploratory Data Analytics**

# In[30]:


df_stress = pd.read_csv('Stress_Nurse.csv')


# In[31]:


df_stress.head(10)


# In[32]:


len(df_stress)


# In[33]:


df_stress = df_stress.rename(columns = {'Step count': 'Step_count', 'Stress Level': 'Stress_Level'})
df_stress.head(5)


# ## **Data Description**
# 

# ### **Humidity**

# In[34]:


df_stress >> group_by(X.Stress_Level) >> mutate(
    Humidity_size = X.Humidity.count(),
    Humidity_min = np.min(X.Humidity),
    Humidity_max = np.max(X.Humidity),
    Humidity_mean = np.mean(X.Humidity),
    Humidity_sd = np.std(X.Humidity),
    Humidity_cil = np.mean(X.Humidity) - 1.96 * np.std(X.Humidity),
    Humidity_cih = np.mean(X.Humidity) + 1.96 * np.std(X.Humidity)
) >> head(5) >> select(
    X.Stress_Level,
    X.Humidity_size, X.Humidity_min, X.Humidity_max,
    X.Humidity_mean, X.Humidity_sd, X.Humidity_cil, X.Humidity_cih
)


# ### **Temperature**

# In[35]:


df_stress >> group_by(X.Stress_Level) >> mutate(
    Temperature_size = X.Temperature.count(),
    Temperature_min = np.min(X.Temperature),
    Temperature_max = np.max(X.Temperature),
    Temperature_mean = np.mean(X.Temperature),
    Temperature_sd = np.std(X.Temperature),
    Temperature_cil = np.mean(X.Temperature) - 1.96 * np.std(X.Temperature),
    Temperature_cih = np.mean(X.Temperature) + 1.96 * np.std(X.Temperature)
) >> head(5) >> select(
    X.Stress_Level,
    X.Temperature_size, X.Temperature_min, X.Temperature_max,
    X.Temperature_mean, X.Temperature_sd, X.Temperature_cil, X.Temperature_cih
)


# ### **Step count**

# In[36]:


df_stress >> group_by(X.Stress_Level) >> mutate(
    Stepcount_size = X.Step_count.count(),
    Stepcount_min = np.min(X.Step_count),
    Stepcount_max = np.max(X.Step_count),
    Stepcount_mean = np.mean(X.Step_count),
    Stepcount_sd = np.std(X.Step_count),
    Stepcount_cil = np.mean(X.Step_count) - 1.96 * np.std(X.Step_count),
    Stepcount_cih = np.mean(X.Step_count) + 1.96 * np.std(X.Step_count)
) >> head(5) >> select(
    X.Stress_Level,
    X.Stepcount_size, X.Stepcount_min, X.Stepcount_max,
    X.Stepcount_mean, X.Stepcount_sd, X.Stepcount_cil, X.Stepcount_cih
)


# ## **Scatter Plot and Boxplot**
# 

# In[37]:


fig, axs = plt.subplots(ncols=2)
sns.scatterplot(data=df_stress,
                x="Stress_Level", y="Step_count", hue=df_stress.Stress_Level.tolist(),
                ax=axs[0])
sns.boxplot(data=df_stress,
            x="Stress_Level", y="Step_count", hue=df_stress.Stress_Level.tolist(),
            ax=axs[1])
plt.show()


# In[38]:


fig, axs = plt.subplots(ncols=2)
sns.scatterplot(data=df_stress,
                x="Stress_Level", y="Temperature", hue=df_stress.Stress_Level.tolist(),
                ax=axs[0])
sns.boxplot(data=df_stress,
            x="Stress_Level", y="Temperature", hue=df_stress.Stress_Level.tolist(),
            ax=axs[1])
plt.show()


# In[39]:


fig, axs = plt.subplots(ncols=2)
sns.scatterplot(data=df_stress,
                x="Stress_Level", y="Humidity", hue=df_stress.Stress_Level.tolist(),
                ax=axs[0])
sns.boxplot(data=df_stress,
            x="Stress_Level", y="Humidity", hue=df_stress.Stress_Level.tolist(),
            ax=axs[1])
plt.show()


# # Analyses
# 

# ## Data Preparation

# In[40]:


df_y = df_stress['Stress_Level']
df_x = df_stress[df_stress.columns[df_stress.columns != 'Stress_Level']]
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, random_state=42)


# In[41]:


train_x.count


# In[42]:


test_x.count


# # **3-1. Decision Tree**

# In[43]:


# Building a learning model
model_dt = tree.DecisionTreeClassifier(max_depth=3, random_state=42)

# Setting the test data for learning
model_dt.fit(train_x, train_y)

# Conducting inference
test_dt = model_dt.predict(test_x)

# Calculating the accuracy
score_dt = model_dt.score(test_x, test_y)
print(f'Accuracy is {score_dt * 100}%.')


# In[44]:


plot_tree(model_dt, feature_names=train_x.columns, class_names=True, filled=True)


# # **3-2. Random Forest**

# ## **Classification**

# In[45]:


# Building a leaening model
model_rf = RandomForestClassifier(max_depth=3, random_state=42)

# Setting the test data for learning
model_rf.fit(train_x, train_y)

# Conduct inference
test = model_rf.predict(test_x)

# Calculating the accuracy
score_rfc = accuracy_score(test_y, test)
print(f'Accuracy is {score_rfc * 100}%.')


# ## **Regression**

# In[46]:


score_reg = model_rf.score(test_x, test_y)
print(f'Accuracy is {score_reg * 100}%.')


# ### **RMSE and R2 Evaluation**

# In[47]:


# Predicting train_x
train_y_pred = model_rf.predict(train_x)

# Predict test_y
test_y_pred = model_rf.predict(test_x)

# RMSE
print('RMSE Train: %.2f, Test: %.2f' % (
        mean_squared_error(train_y, train_y_pred, squared=False), # Train
        mean_squared_error(test_y, test_y_pred, squared=False)    # test
      ))

# R2
print('R^2 Train: %.2f, Test: %.2f' % (
        r2_score(train_y, train_y_pred), # Train
        r2_score(test_y, test_y_pred)    # Test
      ))


# ### **Residual Plot Evaluation**

# In[48]:


# Plotting predictions and residuals in train data
plt.scatter(train_y_pred,
            train_y_pred - train_y,
            c='blue',
            marker='o',
            s=40,
            alpha=0.7,
            label='Train Data')


# Plotting predictions and residuals in test data
plt.scatter(test_y_pred,
            test_y_pred - test_y,
            c='red',
            marker='o',
            s=40,
            alpha=0.7,
            label='Test Data')

# Configurate Style
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-20, xmax=60, lw=2, color='black')
plt.xlim([-20, 60])
plt.ylim([-50, 40])
plt.tight_layout()
plt.show()


# In[ ]:




