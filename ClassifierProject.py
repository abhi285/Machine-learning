
# coding: utf-8

# In[2]:

#sentiment anlyzer for products.
import graphlab


# In[3]:

# read some product review data
products = graphlab.SFrame('G:/amazon_baby.gl')


# In[20]:

#lets explore this data
products.head()


# In[21]:

#word count vector for each review
products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[22]:

products.head()


# In[23]:

#What reviews we have for different product names.
graphlab.canvas.set_target('ipynb')
products['name'].show()


# In[24]:

#explore vulli sophie
giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[25]:

len(giraffe_reviews)


# In[27]:

giraffe_reviews['rating'].show(view='Categorical')


# In[28]:

# build a sentiment classifier
products['rating'].show(view='Categorical')



# In[30]:

# define what is a positive and negative sentiment
#ignore all 3 star review

products= products[products['rating'] !=3]


# In[31]:

# positive sentiment = 4* or 5* reviews
products['sentiment']=products['rating'] >=4


# In[32]:

products.head()


# In[35]:

# ready to train our sentiment classifier
train_data,test_data = products.random_split(.8,seed=0)


# In[36]:

sentiment_model = graphlab.logistic_classifier.create(train_data,target='sentiment',features=['word_count'],validation_set=test_data)


# In[37]:

#Evaluating a classifier/sentiment model & the ROC curve
sentiment_model.evaluate(test_data,metric='roc_curve')


# In[38]:

sentiment_model.show(view='Evaluation')


# In[ ]:




# In[39]:

#Applying model to find most positive & negative reviews for a product
giraffe_reviews['predicted_sentiments'] = sentiment_model.predict(giraffe_reviews,output_type='probability')


# In[40]:

giraffe_reviews.head()


# In[41]:

#sort reviews based on the predicted sentiment ans explore
giraffe_reviews = giraffe_reviews.sort('predicted_sentiments',ascending=False)


# In[42]:

giraffe_reviews.head()


# In[44]:

#Exploring the most positive & negative aspects of a product
giraffe_reviews[1]['review']


# In[45]:

#show most negative reviews
giraffe_reviews[-1]['review']


# In[5]:

#write the function
#products['word_count'][1]['and']

products['word_count'] = graphlab.text_analytics.count_words(products['review'])

def awesome_count(dict1):
    dict1=dict()
    if 'awesome' in dict1:
        count = dict1['awesome']
    else:
        count=0
    return count
    

products['awesome'] = products['word_count'].apply(awesome_count)
        


# In[9]:

import ctypes,inspect,os,graphlab
from ctypes import wintypes
kernel32 = ctypes.WinDLL('Kernel32', use_last_error=True)
kernel32.SetDllDirectoryW.argtypes = (wintypes.LPCWSTR,)
src_dir=os.path.split(inspect.getfile(graphlab))[0]
kernel32.SetDllDirectoryW(src_dir)




# In[10]:

graphlab.SArray(range(1000)).apply(lambda x :x)


# In[ ]:




# In[ ]:




# In[ ]:



