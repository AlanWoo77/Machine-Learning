#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:31:41 2019

@author: chandler; additions by Ed.
"""

# IMDB Clustering exercise to illustrate k-means, LDA, and cosine similarity
# Thanks to Brandon Rose (see bottom for GitHub and personal webpage links)

# NOTES
#   I use BR's scraped data (*.txt), but he's made these scraping scripts avaiable via GitHub
#   BR's original doc is available as a very nice Jupyter notebook. I've just copied/pasted here
#   Link to BR's Jupyter notebook: https://github.com/brandomr/document_cluster/blob/master/cluster_analysis.ipynb
#   Results of BR's clustering are described here: http://brandonrose.org/top100
#   Code and graphical results are here: http://brandonrose.org/clustering

###########
# Install necessary libraries
############

#!pip install beautifulsoup4
#!pip install nltk
#!pip install mpld3

###########
# Import necessary libraries
############
from __future__ import print_function
# Joblib is a set of tools to provide lightweight pipelining in Python.
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from scipy.cluster.hierarchy import ward, dendrogram
from nltk.tag import pos_tag
from sklearn.manifold import MDS
import matplotlib as mpl
import matplotlib.pyplot as plt
import os  # for os.path.basename
import pandas as pd
import nltk  # stands for natural language tool kit
from bs4 import BeautifulSoup
import re
import codecs
from sklearn import feature_extraction
import mpld3

###########
# Import necessary lists
############
nltk.download('stopwords')
nltk.download('punkt')

############
# Read data. Then eliminate stop words, tokenize, and stem

# Note: Set your working directory to our Google Drive folder first, then:
# Mac
Git_Hub_Folder = "https://github.com/AlanWoo77/Machine-Learning.git"
os.chdir(Git_Hub_Folder)  # Or Day 3, 4, 6 or 7...
# PC
# os.chdir('.\Lecture 5\imdb clustering') #Or Day 3, 4, 6 or 7...


############
# import three lists: titles, links and wikipedia synopses
titles = open('title_list.txt').read().split('\n')
# ensures that only the first 100 are read in
titles = titles[:100]

links = open('link_list_imdb.txt').read().split('\n')
links = links[:100]

synopses_wiki = open('synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]

synopses_clean_wiki = []
for text in synopses_wiki:
    text = BeautifulSoup(text, 'html.parser').getText()
    # strips html formatting and converts to unicode
    synopses_clean_wiki.append(text)

synopses_wiki = synopses_clean_wiki


genres = open('genres_list.txt').read().split('\n')
genres = genres[:100]

# do some quick checks
print(str(len(titles)) + ' titles')
print(str(len(links)) + ' links')
print(str(len(synopses_wiki)) + ' synopses')
print(str(len(genres)) + ' genres')

# load the imdb synopses
synopses_imdb = open('synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

synopses_clean_imdb = []

for text in synopses_imdb:
    text = BeautifulSoup(text, 'html.parser').getText()
    # strips html formatting and converts to unicode
    synopses_clean_imdb.append(text)

synopses_imdb = synopses_clean_imdb

##
synopses = []

for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)

# generates index for each item in the corpora (in this case it's just rank) and I'll use this for scoring later
ranks = []

for i in range(0, len(titles)):
    ranks.append(i)

# load nltk's English stopwords as variable called 'stopwords'
# Note that you'll need to execute the 'download' statement the first time you use this (and to keep up-to-date?)
#from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'; Stemming is just the process of breaking a word down into its root.
stemmer = SnowballStemmer("english")

# Tokenizing is the process of splitting a text into a list of its respective words (or tokens).

# BR defines a tokenizer and stemmer which returns the set of stems in the text that it is passed


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(
        text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# Create two vocabularies: one tokenized, one stemmed
# Note: will need to download punkt using this the first time:
    # nltk.download('punkt')
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

# Create a DataFrame with the vocabulary;
# BR says: Using these two lists, I create a pandas DataFrame with the stemmed vocabulary as the index and the tokenized words as the column.
# The benefit of this is it provides an efficient way to look up a stem and return a full token.
# The downside here is that stems to tokens are one to many: the stem 'run' could be associated with 'ran', 'runs', 'running', etc.
# For my purposes this is fine--I'm perfectly happy returning the first token associated with the stem I need to look up.
vocab_frame = pd.DataFrame(
    {'words': totalvocab_tokenized}, index=totalvocab_stemmed)

# Now let's encode with tf-idf

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

# cosine similarity is a method for computing distance between pairs of tfidf vectors
dist = 1 - cosine_similarity(tfidf_matrix)

########
# That was all pre-processing. Now let's cluster
########

# K-means Clustering:
# BR: Now onto the fun part. Using the tf-idf matrix, you can run a slew of clustering algorithms to better understand the hidden structure within the synopses.
# I first chose k-means. K-means initializes with a pre-determined number of clusters (I chose 5). Each observation is assigned to a cluster (cluster assignment) so as to minimize the within cluster sum of squares.
# Next, the mean of the clustered observations is calculated and used as the new cluster centroid.
# Then, observations are reassigned to clusters and centroids recalculated in an iterative process until the algorithm reaches convergence.

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# from sklearn.externals import joblib #This didn't work for some reason.
#joblib.dump(km,  'doc_cluster.pkl')
#km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

films = {'title': titles, 'rank': ranks, 'synopsis': synopses,
         'cluster': clusters, 'genre': genres}
frame = pd.DataFrame(films, index=[clusters], columns=[
                     'rank', 'title', 'cluster', 'genre'])
frame['cluster'].value_counts()

grouped = frame['rank'].groupby(frame['cluster'])
grouped.mean()

# What terms are associated with what cluster? Print some titles
print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[
              0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d titles:" % i, end='')
    for title in frame.loc[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()
    print()

# multi-dimensional scaling
MDS()
# two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]

# strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text


def strip_proppers_POS(text):
    tagged = pos_tag(text.split())  # use NLTK's part of speech tagger
    non_propernouns = [word for word,
                       pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns


# Visualize the Clusters
# set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02',
                  2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
# set up cluster names using a dict
'''cluster_names = {0: 'Family, home, war', 
                 1: 'Police, killed, murders', 
                 2: 'Father, New York, brothers', 
                 3: 'Dance, singing, love', 
                 4: 'Killed, soldiers, captain'}
'''  # Our clusters were different. So we redefine them.
cluster_names = {0: 'Ship, army, family, escape',
                 1: 'Car, police, killed, father, friends',
                 2: 'Marries, family, brothers, typ. male pro/anta-gonists',
                 3: 'Love, relationship, children, new',
                 4: 'Soldiers, captain, war'}
%matplotlib inline
# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

# group by cluster
groups = df.groupby('label')

# set up plot (execute in one go til plt.show)
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(
        axis='y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=8)
plt.show()  # show the plot

# uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)

#########
# Hierarchical Clustering
##########
# define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix = ward(dist)
fig, ax = plt.subplots(figsize=(15, 20))  # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
plt.tight_layout()  # show plot with tight layout
# uncomment below to save figure
# plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

#################
# END HERE
#################
'''

#############
#LDA
#############

#strip any proper names from a text...unfortunately right now this is yanking the first word from a sentence too.
import string
def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
from nltk.tag import pos_tag
def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns


#Latent Dirichlet Allocation implementation with Gensim
from gensim import corpora, models, similarities 
#remove proper names
preprocess = [strip_proppers(doc) for doc in synopses]
%time tokenized_text = [tokenize_and_stem(text) for text in preprocess]
%time texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

#print(len([word for word in texts[0] if word not in stopwords]))
print(len(texts[0]))

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in texts]
len(corpus)

%time lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, update_every=5, chunksize=10000, passes=100)
print(lda[corpus[0]])
topics = lda.print_topics(5, num_words=20)
topics_matrix = lda.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix)
#Matrix is wrong shape
topics_matrix.shape
topic_words = topics_matrix[:,:,1]
for i in topic_words:
    print([str(word) for word in i])
    print()
'''

# https://github.com/brandomr/document_cluster
# http://brandonrose.org/clustering
