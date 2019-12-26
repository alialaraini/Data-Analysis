
# coding: utf-8

# In[1]:


#Ali Alaraini
#Project 4: Ridge Regression 


# In[7]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


# In[8]:


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
            'sqft_living15':float, 'grade':int, 'yr_renovated':int,
            'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
            'sqft_lot15':float, 'sqft_living':float, 'floors':float,
            'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
            'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# In[13]:


sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(['sqft_living','price'])



# In[18]:


set1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)


# In[19]:


l2_penalty = 1.5e-5
l2_penalty2 = 1e-9
l2_penalty3 = 1.23e2


# In[20]:


def polynomial_dataframe(feature, deg):
    data_frame = pd.DataFrame()
    data_frame['x1'] = feature
    for i in range (1, deg):
        name='x'+str(i+1)
        data_frame[name] = data_frame['x'+str(i)]*data_frame['x1']
    return data_frame
def polynomial_regression(df,y,penalty):
    model=linear_model.Ridge(alpha=penalty, normalize=True)
    model.fit(df,y)
    return model


# In[31]:


def coef (model):
    deg = len(model.coef_)
    w = list(model.coef_)
    
    print ('polynomial degree' + str(deg) + '='+'\n')
    w.reverse()
    w.append(model.intercept_)
    print(np.poly1d(w))
    
def poly_and_graph(x, y, deg, penalty):
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    df = polynomial_dataframe(x, deg)
    model = polynomial_regression(df, y, penalty)
    
    coef(model)
    graph = plt.plot(df['x1'],y,'.',df['x1'],model.predict(df),'-')
    plt.show(graph)


# In[32]:


poly_and_graph(sales['sqft_living'],sales['price'], 15, l2_penalty)


# In[33]:


Quiz4: 124.9


# In[35]:


poly_and_graph(set1['sqft_living'],set1['price'], 15, l2_penalty2)


# In[36]:


poly_and_graph(set2['sqft_living'],set2['price'], 15, l2_penalty2)


# In[37]:


poly_and_graph(set3['sqft_living'],set3['price'], 15, l2_penalty2)


# In[41]:


poly_and_graph(set4['sqft_living'],set4['price'], 15, l2_penalty2)


# In[38]:


poly_and_graph(set1['sqft_living'],set1['price'], 15, l2_penalty3)


# In[39]:


poly_and_graph(set2['sqft_living'],set2['price'], 15, l2_penalty3)


# In[40]:


poly_and_graph(set3['sqft_living'],set3['price'], 15, l2_penalty3)


# In[42]:


poly_and_graph(set4['sqft_living'],set4['price'], 15, l2_penalty3)


# In[ ]:


#smallest set = set 4 = 2.086
#Largest set = set 1 = 2.328

