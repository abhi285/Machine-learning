
# coding: utf-8

# In[24]:

import graphlab


# In[25]:

# load music data
songdata = graphlab.SFrame('C:/Users/Abhijith Antony/Downloads/song_data.gl')


# In[26]:

# exploring of music data
songdata.head()


# In[27]:

graphlab.canvas.set_target('ipynb')


# In[28]:

songdata['song'].show()


# In[29]:

len(songdata)


# In[30]:

#count number of users
users=songdata['user_id'].unique()
len(users)


# In[31]:

# create a recommender system
traindata,testdata = songdata.random_split(.8,seed=0)


# In[32]:

# simple popularity based recommender
popularitymodel= graphlab.popularity_recommender.create(traindata,user_id='user_id',item_id='song')


# In[33]:

# use the popularity model to make some predctions
popularitymodel.recommend(users=[users[0]])


# In[34]:

# use the popularity model to make some predctions
popularitymodel.recommend(users=[users[1]])
# everyone will be recommended with the same songs which is an issue with the model.


# In[17]:

# creating a personalized recommender model
personalizedmodel= graphlab.item_similarity_recommender.create(traindata, user_id='user_id', item_id='song')


# In[35]:

# applying personalized model to make song recommendation 

personalizedmodel.recommend(users=[users[0]])


# In[36]:

personalizedmodel.recommend(users=[users[1]])


# In[37]:

personalizedmodel.get_similar_items(['With Or Without You - U2'])


# In[39]:

personalizedmodel.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])


# In[42]:

#quantitative comparison between the models
#precision recall curve for these 2 models.
get_ipython().magic(u'matplotlib inline')
modelperformance = graphlab.recommender.util.compare_models(testdata,[popularitymodel,personalizedmodel],user_sample=0.05)


# In[41]:




# In[ ]:



