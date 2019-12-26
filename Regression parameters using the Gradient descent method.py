
# coding: utf-8

# In[56]:


import numpy as np
from matplotlib import pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


x_points = [0,1,2,3,4]
y_points = [1,3,7,13,21]


# In[14]:


plt.plot(x_points,y_points,'bo')


# In[71]:


# y = mx + b 
step_size = 0.05
m = 0
b = 0
tolerance = 0.01
y = lambda x : m*x + b


# In[72]:


def plot_line (y, data_points):
    x_values = [i for i in range(int(min(data_points))-1, int(max(data_points))+2)]
    y_values = [y(x) for x in x_values]
    plt.plot(x_values, y_values, 'r')


# In[82]:


plot_line(y,x_points)
plt.plot(x_points, y_points, 'bo')


# In[78]:


def summation(y, x_points, y_points):
    total1 = 0
    total2 = 0
    
    for i in range(1, len(x_points)): 
        total1 += y(x_points[i])-y_points[i]
        total2 += (y(x_points[i])- y_points[i])*x_points[i]
    return total1 / len(x_points), total2 / len(x_points)


# In[88]:


for i in range (50):
    s1, s2 = summation(y, x_points, y_points)
    m = m - learn * s2
    b = b - learn * s1
    plot_line(y,x_points)
    plt.plot(x_points, y_points, 'bo')


# In[85]:


m


# In[86]:


b


# In[87]:


plot_line(y,x_points)
plt.plot(x_points, y_points, 'bo')

