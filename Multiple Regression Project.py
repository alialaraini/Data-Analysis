
# coding: utf-8

# In[1]:


# Ali Alaraini
#Project 2: Multiple Regression 
#April 9th 2018


# In[4]:


import pandas as pd
import numpy as np


# In[5]:


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':str, 
              'condition':int, 'lat':float, 'date':str, 
              'sqft_basement':int, 'yr_built':int, 'id':str, 
              'sqft_lot':int, 'view':int}


# In[6]:


#load data of houses sale 
sales = pd.read_csv('kc_house_data.csv')


# In[7]:


#split data into training and testing 
train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')


# In[8]:


train_data[train_data['bathrooms']==np.nan]


# In[9]:


#Learning a multiple regression model
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)


# In[10]:


train_data['bed_bath_rooms'] = train_data.apply(lambda x:x['bedrooms']*x['bathrooms'],axis=1)
test_data['bed_bath_rooms'] = test_data.apply(lambda x:x['bedrooms']*x['bathrooms'],axis=1)
train_data['log_sqft_living'] = train_data.apply(lambda x: np.log(x['sqft_living']),axis=1)
test_data['log_sqft_living'] = test_data.apply(lambda x: np.log(x['sqft_living']),axis=1)
train_data['lat_plus_long'] = train_data.apply(lambda x: x['lat'] + x['long'],axis=1)
test_data['lat_plus_long'] = test_data.apply(lambda x: x['lat'] + x['long'],axis=1)





# In[11]:


#Averges of 4 test data


# In[12]:


round(test_data['bedrooms_squared'].mean())


# In[13]:


round(test_data['bed_bath_rooms'].mean(),2)


# In[14]:


round(test_data['log_sqft_living'].mean(),2)


# In[15]:


round(test_data['lat_plus_long'].mean(),2)


# In[16]:


#Creating Multiple Models
model_1 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2 = model_1 + ['bed_bath_rooms']
model_3 = model_2 + ['bedrooms_squared','log_sqft_living','lat_plus_long']


# In[17]:


import sklearn as sk
from sklearn import linear_model


# In[18]:


model1 = sk.linear_model.LinearRegression()
model1.fit(train_data[model_1],train_data['price'])



# In[19]:


model2 = sk.linear_model.LinearRegression()
model2.fit(train_data[model_2],train_data['price'])



# In[20]:


model3 = sk.linear_model.LinearRegression()
model3.fit(train_data[model_3],train_data['price'])


# In[21]:


model1.coef_


# In[22]:


model2.coef_


# In[23]:


model3.coef_


# In[ ]:


#Quiz 6?
#Psositive 
#Quiz 7?
#negative
#Quiz 8? why different
#models did change their signs because the relationship between the indepedent variables and depedent variable varies between the three models. 


# In[38]:


def get_residual_sum_of_squares(model, data, outcome):
    
    prediction = model.predict(data)

    residual = outcome - prediction

    RSS = (residual ** 2).sum()
    
    return(RSS)


# In[40]:


rss_train_model1 = get_residual_sum_of_squares(model1, train_data[model_1], train_data['price'])
print (rss_train_model1)


# In[41]:


rss_train_model2 = get_residual_sum_of_squares(model2, train_data[model_2], train_data['price'])
print (rss_train_model2)


# In[42]:


rss_train_model3 = get_residual_sum_of_squares(model3, train_data[model_3], train_data['price'])
print (rss_train_model3)


# In[ ]:


#Quiz 9?
# model 3 had the lowest RSS which means that it was the best fit model out of the 3 since it has more features


# In[43]:


rss_test_model1 = get_residual_sum_of_squares(model1, test_data[model_1], test_data['price'])
print (rss_test_model1)


# In[44]:


rss_test_model2 = get_residual_sum_of_squares(model2, test_data[model_2], test_data['price'])
print (rss_test_model2)


# In[45]:


rss_test_model3 = get_residual_sum_of_squares(model3, test_data[model_3], test_data['price'])
print (rss_test_model3)


# In[ ]:



#model 2 had the lowest RSS


# In[ ]:



# since we used different data sets, the calculated RSS was different. This is because the test data measure the quality of performance of the at making predictions on that test set. 

