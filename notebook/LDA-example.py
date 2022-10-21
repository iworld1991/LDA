# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Latent Dirichlet Allocation (LDA): An Ilustrative Example
#
# - author: Tao Wang
# - I refer to following online resources:
#   - [Topic Modeling in Python: Latent Dirichlet Allocation (LDA)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjKnJr92s3zAhUhpnIEHQ7iBicQFnoECA4QAQ&url=https%3A%2F%2Ftowardsdatascience.com%2Fend-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0&usg=AOvVaw3BOumfnMly0lUh3pfEHrWd)
#   -[Topic Modeling with Gensim (Python) - Machine Learning Plus](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjKnJr92s3zAhUhpnIEHQ7iBicQFnoECAIQAQ&url=https%3A%2F%2Fwww.machinelearningplus.com%2Fnlp%2Ftopic-modeling-gensim-python%2F&usg=AOvVaw09WN-93Y-Jk0fbq3KWF7qF)
#   - [Topic modeling visualization – How to present the results of LDA models?](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
#   

# +
import time

start_code = time.time()
# -

Fast = True
Prange = True

# ## Data importing

import pandas as pd

# +
## Here, we use a sub-sample of NYT articles containing key words "inflation" as an example 
## it is saved as a pickle file (a handy and light-weighted data format in python) 
## the path and name

file_name = '../data/article_data.pkl'

article_data = pd.read_pickle(file_name)
# -

## an example of the article 
## the second article in the database and the first 1000 words in the article
## only print part of the article to save the space 
article_data.iloc[2]['text'][:1000]

# ## Text Preprocessing 
#
# We will perform the following steps:
#
# - __Tokenization__: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
# - Words that have fewer than 2 characters are removed.
# - All __stopwords__ are removed.
# - Words are __lemmatized__ — words in third person are changed to first person and verbs in past and future tenses are changed into present.
# - Words are __stemmed__ — words are reduced to their root form.

# # I did not have gensim package installed. 
# # so the following code is used to install it
# # comment it out once it is run once 


import re
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
#import nltk.stem as stemmer
import numpy as np
import nltk
import matplotlib.pyplot as plt
#nltk.download('wordnet')

from wordcloud import WordCloud


# ## stemming and lemmatizing
#
# - Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. 
# - Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma .

# + code_folding=[0]
## write a function to lemmatize and stem the words 

def remove_email(text):
    return [re.sub('\S*@\S*\s?', '', email) for email in text]

def lemmatize_stemming(word):
    stemmer = SnowballStemmer("english")
    return WordNetLemmatizer().lemmatize(stemmer.stem(word))

## also remove stop words
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token)>=2:
            result.append(lemmatize_stemming(token))  
    return result


# + code_folding=[0]
## apply above functions to an example article 

random_article = np.random.randint(0,1000)

one_article = article_data.iloc[random_article]['text']

print('original document: ')
print(one_article)

print('splitted into words: ')
one_artile = remove_email(one_article)
tokens = gensim.utils.simple_preprocess(one_article)
#words = []
#for word in one_article.split():
#    words.append(word)
print(tokens[:100])
print('\n\n tokenized and lemmatized document: ')
words_processed = preprocess(one_article)
print(words_processed[:200])
# -

# ### Generate a wordcloud for an example article

# + code_folding=[1]
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", 
                      max_words=500, 
                      contour_width=3, 
                      contour_color='steelblue')
# Generate a word cloud
long_string=(" ").join(words_processed)  
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()

# + code_folding=[0]
## process all articles



start_process = time.time()

all_tokens = {}  # empty dictionary to save
id_map = {} ## map the id in the dataset to the id in the matrix (some articles are empty
id_count = -1  


if Prange:
    from numba import prange
    for i in prange(len(article_data)):
        #print(i)
        this_article = article_data['text'].iloc[i]
        if type(this_article)==str:
            #print('article '+str(i) + ' works')
            processed_docs = preprocess(this_article)
            all_tokens[i] = processed_docs
            #print(processed_docs[:10])
            id_count+=1
            id_map[i]=id_count 
            #print(id_count)

else:
    for i in range(len(article_data)):
        #print(i)
        this_article = article_data['text'].iloc[i]
        if type(this_article)==str:
            #print('article '+str(i) + ' works')
            processed_docs = preprocess(this_article)
            all_tokens[i] = processed_docs
            #print(processed_docs[:10])
            id_count+=1
            id_map[i]=id_count 
            #print(id_count)

all_tokens_list = [all_tokens[i] for i in all_tokens.keys()]

end_process = time.time()

# +
print('Here are an example of the preprocessed words from a particular article')

print(all_tokens_list[10])
# -

# ## Bag of Words on the Dataset
#
# Create a dictionary from ‘processed_docs’ containing the number of times a word appears in the training set.
#

# +
from gensim.models import Phrases
bigram = Phrases(all_tokens_list, min_count=20)

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
for idx in range(len(all_tokens_list)):
    for token in bigram[all_tokens_list[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            all_tokens_list[idx].append(token)

# +
## get the dictionary over all articles

dictionary = gensim.corpora.Dictionary(all_tokens_list)

## filter extreme values 
nb_doc = len(all_tokens_list)

no_below =int(0.01*nb_doc)
no_above =int(0.8*nb_doc)

dictionary.filter_extremes(no_below = no_below,
                          no_above = no_above)

# + code_folding=[3]
## print some words from the dictionary 



print('The length of the dictionary: '+str(len(dictionary)))
print('Here are some words from the dictionary:')
count = 0
for x in dictionary.values():
    count=count + 1
    if count <=20:
        print(x)
    
# -

print("the corpus is essentially a frequency count of the present temrs in each article")

## generate a bag of words for each article 
## i.e. count the frequency of each word in the dictionary in each article 
corpus = [dictionary.doc2bow(i) for i in all_tokens_list]

# ## Initialize LDA model
#
#

# - The corpus or the document-term matrix to be passed to the model (in our example is called tokens_matrix)
# - Number of Topics: num_topics is the number of topics we want to extract from the corpus.
# - id2word: It is the mapping from word indices to words. Each of the words has an index that is present in the dictionary.
# - Number of Iterations: it is represented by Passes in Python. Another technical word for iterations is ‘epochs’. Passes control how often we want to train the model on the entire corpus for convergence.
# - Chunksize: It is the number of documents to be used in each training chunk. The chunksize controls how many documents can be processed at one time in the training algorithm.
#    - Alpha: is the document-topic density
#    - Beta: (In Python, this parameter is called ‘eta’): is the topic word density
#  
#     - For instance, the higher values of alpha —> the documents will be composed of more topics, and
#     - The lower values of alpha —> returns documents with fewer topics.

# + code_folding=[]
import time 

start_lda = time.time()

Lda = gensim.models.ldamodel.LdaModel
LdaM = gensim.models.ldamulticore.LdaMulticore


nb_topics = 3

if Fast:
    ldamodel = LdaM(corpus,
                   num_topics= nb_topics,
                   id2word=dictionary,
                   chunksize = 100,
                   alpha='asymmetric', ## or 'symmetric'
                   eta='auto',
                   iterations = 200,
                   passes=1,
                   random_state=2019,
                   eval_every=None)
    
else:
    ldamodel = Lda(corpus,
                   num_topics= nb_topics,
                   id2word=dictionary,
                   chunksize = 100,
                   alpha='auto', ## or 'symmetric'
                   eta='auto',
                   iterations = 200,
                   passes= 1,
                   random_state=2019,
                   eval_every=None)
    
## save the model as an instance so that no need to retrained everytime 
ldamodel.save('./model/trained_results.model')
end_lda = time.time()

print('time taken to run the lda model: is {}'.format(str(end_lda-start_lda)))

# +
## load the model from storage 

ldamodel = gensim.models.ldamodel.LdaModel.load('./model/trained_results.model')
# -

print('These are the the most common words for each topic')
ldamodel.print_topics(num_words=20)

# ### Assign topics to each article 
#

count = 0 
for i in ldamodel[corpus]:
    if count <=30:
        print('article',count,i)
    count +=1 

# - For instance, article 1 has the highest weight (0.8411557) on the second topic 

# ### Assign topics to each article and creating article-weight matrix

# +
print('nb of articles in the sample: '+str(len(article_data)))
print('nb of articles with non-empty topic model results: '+str(len(ldamodel[corpus])))
print('\n\n')

count = 0 
for i in ldamodel[corpus]:
    if count <=30:
        print('article',count,i)
    count +=1 

# +
article_weight_dict = {}

for count, topic_weight in enumerate(ldamodel[corpus]):
    #print(count)
    #print(topic_weight)
    this_article_weight = dict(topic_weight)
    #print(this_article_weight)
    article_weight_dict.update({count:this_article_weight})
# -

## convert it to a dataframe 
article_weight = pd.DataFrame.from_dict(article_weight_dict,orient='index')
article_weight = article_weight.fillna(0.0)

# +
## each article and its topic weight

article_weight

# +
## representative article 

## rep article

rep_article_dict = {}
rep_article_text_dict = {}

for i in range(nb_topics):
    ## article id
    rep_id = article_weight[i].argmax()
    rep_article_dict.update({i:rep_id})
    
    ## articlde text
    rep_text = article_data['text'].iloc[rep_id]
    rep_article_text_dict.update({i:rep_text})
    
rep_article = pd.DataFrame.from_dict(rep_article_dict,
                                     orient='index',
                                    columns=['article_id'])

rep_article_text = pd.DataFrame.from_dict(rep_article_text_dict,
                                     orient='index',
                                    columns=['article_text'])

rep_article = pd.merge(rep_article,
                      rep_article_text,
                      left_index =True,
                      right_index=True)

rep_article
# -

# ## Visualization of the topic models 
#

# + code_folding=[]
## uncomment the code below if there is no pyLDAvis installed 
#pip install pyLDAvis
# -

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook()
vis = gensimvis.prepare(ldamodel, corpus, dictionary=ldamodel.id2word)
pyLDAvis.save_html(vis, '../figure/first_run.html')   ## save it as a html file 
vis   ## show the figure 

end_code = time.time()

# ## Topic intensity over time

# + code_folding=[2]
## in the main dataset, we add columns sized of nb of topics, recording the score of each topic of that article 

for nb in range(nb_topics):
    
    ## for a particular topic 
    weight_dict = {}
    
    for i in range(len(article_data)):
        if i in id_map.keys():
            this_id = id_map[i]  ## id_map maps id in the dataset and in the model 
            weight_list = dict(ldamodel[corpus][this_id])
            
            
            if nb in weight_list:
                #article_data['weight_topic'+str(nb)].iloc[i] = weight_list[nb]
                weight_dict[i] = weight_list[nb]
            else:
                #article_data['weight_topic'+str(nb)].iloc[i] = 0.0
                weight_dict[i] = 0.0
                
    # merge this back to main data
    weight_df = pd.DataFrame(list(weight_dict.items()),
                             columns = ['id','weight_topic'+str(nb)])
    weight_df.set_index('id', drop = True, inplace = True)
    article_data = pd.merge(article_data,
                           weight_df,
                           left_index = True,
                           right_index = True,
                           how='outer')

# +
## the columns on the right are newly added 

article_data.head(5)
# -

## set date 
article_data['date'] = pd.to_datetime(article_data['date'],
                                      errors='coerce')

import datetime as dt 
article_data['month_date'] = pd.to_datetime(article_data['date']).dt.to_period('M')

# + code_folding=[0]
## day by day 
fig = plt.figure(figsize=(20,5))
for nb in range(nb_topics):
    intensity = article_data.groupby(['date'])['weight_topic'+str(nb)].mean()
    intensity_mv = intensity.rolling(7).mean()
    intensity_mv.plot(lw=3,
                   style='--',
                   label='topic'+str(nb+1))
plt.legend(loc=0)
plt.title('Average Topic Intensity over time (7-day moving average)')
## notice in this data, the dates are very sparse, hence daily plot may not be very meaningful 

# + code_folding=[0]
## month by month 

fig = plt.figure(figsize=(10,5))
for nb in range(nb_topics):
    intensity = article_data.groupby(['month_date'])['weight_topic'+str(nb)].mean()
    #intensity = intensity/intensity[0]
    intensity.plot(lw=4,
                   style='--',
                  label='topic'+str(nb+1))
    #plt.plot(intensity,
    #         lw = 2,
    #         label='topic'+str(nb+1))
plt.legend(loc=0)
plt.title('Average Topic Intensity over time (monthly)')
# -
# ## Run time

print('time taken to process the text: is {}'.format(str(end_process-start_process)))
print('time taken to train the lda model: is {}'.format(str(end_lda-start_lda)))
print('time taken to run all code : is {}'.format(str(end_code-start_code)))


