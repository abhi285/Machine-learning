
# coding: utf-8

# In[1]:

import graphlab


# In[2]:

# load some house sales data
sales = graphlab.SFrame('G:\home_data.gl')


# In[3]:

sales[]


# In[7]:

#exploring data for housing
graphlab.canvas.set_target('ipynb') #ipynotbook and not browser
sales.show(view="Scatter Plot",x="sqft_living",y="price")


# In[8]:

#create simple regression model of square foot of living to price.
# first thing Splitting the data into training and test sets

train_data,test_data = sales.random_split(.8,seed=0) #use  80 %for training and 20 % for test set seed to it


# In[9]:

#build regression model use premade algorithm
sqft_model =graphlab.linear_regression.create(train_data,target='price',features=['sqft_living']) #in the input i give the training data
#if we dont give features all are used as param


# In[10]:

#Evaluating error (RMSE) of the simple model (now we have trained a linear regression model, we will use test data)
print test_data['price'].mean()


# In[12]:

print sqft_model.evaluate(test_data) # one was an outlier with max error we can see


# In[15]:

#Visualizing predictions of simple model with Matplotlib
#lets show what our predictions look like
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[16]:

# build matplotlib for results
plt.plot(test_data['sqft_living'],test_data['price'],'.', 
        test_data['sqft_living'],sqft_model.predict(test_data),'.') #each point as dot


# In[21]:

#Inspecting the model coefficients learned
sqft_model.get('coefficients') #intercept is where the line crosses the y axis, price/sq feet : 280$ per sq ft average for seattle.


# In[25]:

#Exploring other features of the data, had many other columns associated with it
my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']


# In[26]:

sales[my_features].show()


# In[27]:

sales.show(view='BoxWhisker Plot', x='zipcode',y='price') #98004 has highest average and huge variability, 98039


# In[28]:

#Learning a model to predict house prices from more features
my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features)


# In[29]:

print my_features


# In[31]:

# now we square foot model and the my features model now we will compare these 2.
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data) # max error and rmse went down for features model, add more and more less error. more they can learn.


# In[32]:

#Applying learned models to predict price of an average house for 3 house.
house1 = sales[sales['id']=='5309101200']


# In[33]:

house1


# In[35]:

#also html and images
#<img src="path"
print house1['price']


# In[37]:

print sqft_model.predict(house1)


# In[38]:

print my_features_model.predict(house1) #even though on  average more features give better prediction for this one house not the case.


# In[39]:

#Applying learned models to predict price of two fancy houses for a second house
house2 =sales[sales['id']=='1925069082']


# In[40]:

house2


# In[42]:

print my_features_model.predict(house2) # standard house fare bad, but with features fare better (more bathrooms bedrooms).


# In[43]:

print sqft_model.predict(house2)


# In[51]:

#assignments:
#1. for zip code : 98039 find average

sales['zipcode'=='98039']
sales

