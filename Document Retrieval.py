
# coding: utf-8

# In[1]:

import graphlab


# In[4]:

# Load some text data from wikipedia pages on people
people =graphlab.SFrame('people_wiki.gl/')



# In[5]:

people.head()


# In[6]:

len(people)


# In[11]:

#exploring dataset checkout the text it contains
obama = people[people['name']== 'Barack Obama']
obama['text']

clooney = people[people['name']== 'George Clooney']
clooney['text']


# In[15]:

# exploring word counts
obama['Word_Count'] = graphlab.text_analytics.count_words(obama['text'])


# In[19]:

obama.head()
#sort the word count for the obama ar.sort(ticle (word count is a dictionary)
obama_word_count_table = obama[['Word_Count']].stack('Word_Count',new_column_name=['word','count'])
obama_word_count_table.head()


# In[20]:

obama_word_count_table.sort('count',ascending=False)


# In[23]:

#computing TF IDF's for the corpus
people['word_count'] = graphlab.text_analytics.count_words(people['text'])


# In[34]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
tfidf


# In[32]:

people['tfidf'] = tfidf


# In[33]:

people.head()


# In[35]:

# examine tf idf for the obama article
obama= people[people['name']=='Barack Obama']


# In[38]:

obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# In[41]:

# manually computing distance between a few people
clinton=people[people['name']=='Bill Clinton']
beckham=people[people['name']=='David Beckham']


# In[42]:

# is obama closer to clinton than to beckham
graphlab.distances.cosine (obama['tfidf'][0],clinton['tfidf'][0])
# cosine similarity higher the number the more similar they are(distance version, lower the better)


# In[43]:

graphlab.distances.cosine (obama['tfidf'][0],beckham['tfidf'][0])


# In[44]:

# lower the better so obama more similar to clinton
# document retrieval using nearest model
knn_model=graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')


# In[45]:

# applying nearest neighbour model for retrieval
# who is closest to obama
knn_model.query(obama)


# In[48]:

#other examples of document retrieval
swift= people[people['name']=='Taylor Swift']


# In[49]:

knn_model.query(swift)


# In[63]:

elton= people[people['name']=='Elton John']
elton


# In[66]:

elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# In[69]:

elton[['word_count']].stack('word_count',new_column_name=['word','count']).sort('count',ascending=False)


# In[70]:

victoria=people[people['name']=='Victoria Beckham']
paul=people[people['name']=='Paul McCartney']


# In[71]:

knn_model.query(elton)


# In[74]:

graphlab.distances.cosine (elton['tfidf'][0],victoria['tfidf'][0])


# In[75]:

graphlab.distances.cosine (elton['tfidf'][0],paul['tfidf'][0])


# In[78]:

knn_model2=graphlab.nearest_neighbors.create(people,features=['word_count'],label='name')


# In[79]:

knn_model2.query(elton)


# In[80]:

knn_model2.query(victoria)


# In[ ]:

graphlab.distances.cosine (elton['tfidf'][0],paul['tfidf'][0])


# In[89]:

billy=people[people['name']=='Billy Joel']
cliff=people[people['name']=='Cliff Richard']
roger=people[people['name']=='Roger Daltry']
george=people[people['name']=='George Bush']
rod=people[people['name']=='Rod Stewart']
tommy=people[people['name']=='Tommy Haas']
elvis=people[people['name']=='Elvis Presley']
elvis


# In[90]:

graphlab.distances.cosine (elton['word_count'][0],billy['word_count'][0])


# In[91]:

graphlab.distances.cosine (elton['word_count'][0],cliff['word_count'][0])


# In[92]:

graphlab.distances.cosine (elton['word_count'][0],roger['word_count'][0])


# In[93]:

graphlab.distances.cosine (elton['word_count'][0],george['word_count'][0])


# In[86]:

graphlab.distances.cosine (elton['tfidf'][0],rod['tfidf'][0])


# In[87]:

graphlab.distances.cosine (elton['tfidf'][0],tommy['tfidf'][0])


# In[88]:

graphlab.distances.cosine (elton['tfidf'][0],elvis['tfidf'][0])


# In[ ]:



