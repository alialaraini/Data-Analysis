
# coding: utf-8

# In[1]:


#ali Alaraini
#April 25th 2018
#Project 3: Multiple Regression 


# In[228]:


import pandas as pd 
import numpy as np


# In[10]:


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':str, 
              'condition':int, 'lat':float, 'date':str, 
              'sqft_basement':int, 'yr_built':int, 'id':str, 
              'sqft_lot':int, 'view':int}



# In[11]:


#split data into training and testing 
train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')


# In[8]:


#load data of houses sale 
sales = pd.read_csv('kc_house_data.csv')


# In[81]:


def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1
    features = ['constant'] + features
    features_frame = data_frame[features]
    feature_matrix = np.array(features_frame)
    output_array = np.array(data_frame[output])
    return(feature_matrix, output_array)


# In[82]:


(example_features, example_output) = get_numpy_data(sales,['sqft_living'],['price'])
example_features[0,:]


# In[83]:


example_output[0]



# In[84]:


my_weights = np.array([1.,1.])
my_features = example_features[0,]
predicted_value =np.dot(my_features, my_weights)
predicted_value


# In[85]:


def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)


# In[86]:


test_predictions = predict_output(example_features, my_weights)
test_predictions[0]


# In[87]:


test_predictions[1]


# In[88]:


#derivative computation 
def feature_derivative(errors, feature): 
    derivative = np.dot(errors, feature)*2
    return(derivative)


# In[133]:


(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([0., 0.])
test_predictions = predict_output(example_features, my_weights)
errors = - example_output
feature = example_features[:,0]
derivative = feature_derivative(errors, feature)

-np.sum(example_output)*2


# In[134]:


derivative


# In[135]:


#gradient descent 
from math import sqrt


# In[136]:


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights)
    while not converged:
        predictions = predict_output(feature_matrix, weights)
        
        errors = predictions - output
        gradient_sum_squares = 0  
        for i in range(len(weights)): # loop over each weight
            feature = feature_matrix[:,i]
            derivative = feature_derivative(errors, feature)
            gradient_sum_squares =  gradient_sum_squares + derivative**2
            weights[i] = weights[i] - step_size * derivative
            
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
        


# In[226]:


simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7


# In[227]:


simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)
simple_weights[1]


# In[140]:


(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)


# In[141]:


predictions = predict_output(test_simple_feature_matrix, simple_weights)


# In[142]:


#Question: What is the value of the weight for sqft_living -- the second element of ‘simple_weights’ (rounded to 1 decimal place)?


# In[143]:


predictions[0]


# In[144]:


error = test_output - predictions
SS = np.dot(error,error)
RSS = sqrt(SS)
RSS


# In[179]:


#running a multiple regression 
model_features = ['sqft_living','sqft_living15'] 
#sqft_living15 is a 15 neighborhood average squarefeet 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9


# In[186]:


multiple_weights = regression_gradient_descent(feature_matrix, output,initial_weights, step_size,tolerance)


# In[202]:


(test_multiple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
predictions = predict_output(test_multiple_feature_matrix, multiple_weights)


# In[203]:


#Quiz Question: What is the predicted price for the 1st house in the TEST data set for model 2 (round to nearest dollar)?


# In[205]:


predictions[0]


# In[206]:


test_output[0]


# In[207]:


#Quiz Question: Which estimate was closer to the true price for the 1st house on the TEST data set, model 1 or model 2?


# In[219]:


error = test_output - predictions
SS = np.dot(error,error)
RSS = sqrt(SS)
RSS


# In[220]:


#Quiz Question: Which model (1 or 2) has lowest RSS on all of the TEST data?


# In[225]:


#model = 2

