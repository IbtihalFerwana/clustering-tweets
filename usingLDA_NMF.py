#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import numpy as np
from sklearn.externals import joblib
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.manifold import MDS
from sklearn import metrics
import matplotlib.pyplot as plt1

from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

tweets = []
# reding from a file
text_file=io.open("combined_tweets_21_12.txt","r",encoding="UTF-8-sig")
lines = text_file.read().split('***')
for ln in lines:
    tweets.append(ln)
# count tf
tf_vectorizer=CountVectorizer(max_df=0.95,min_df=2,max_features=1000)
tf=tf_vectorizer.fit_transform(tweets) #Xfitted
tf_features_names=tf_vectorizer.get_feature_names()

# count tf-idf
tfidf_vectorizer=TfidfVectorizer(max_df=0.95,min_df=2)
tfidf=tfidf_vectorizer.fit_transform(tweets)
tfidf_feature_names=tfidf_vectorizer.get_feature_names()

print 'shape *******'
print tf.shape
dist = 1 - cosine_similarity(tfidf)
print

no_topics=2

# Start Clustering #
lda=LatentDirichletAllocation(n_topics=no_topics,max_iter=100,learning_method='online',learning_offset=50.,random_state=0).fit(tf)
nmf=NMF(n_components=no_topics, random_state=1,alpha=.1,l1_ratio=.5,init='nndsvd').fit(tfidf)
# print top tf-idf words #
def display_topics(H,W,feature_names,documents,no_top_words, no_top_documents):
    for topic_idx,topic in enumerate(H):
        print "Cluster %d: " %(topic_idx)
        print "".join([feature_names[i]+"\n"
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        print
        top_doc_indicies=np.argsort(W[:,topic_idx])[::-1][0:no_top_documents]
        for doc_index in top_doc_indicies:
            print documents[doc_index]

no_top_words=6
no_top_documents= 4

lda_W=lda.transform(tf)
lda_H=lda.components_

#lda_H/lda.components_.sum(axis=1)[:,np.newaxis]
nmf_W=nmf.transform(tfidf)
nmf_H=nmf.components_
print "nmf params"
print lda_W.shape

print "LDA"
display_topics(lda_H,lda_W,tf_features_names,tweets,no_top_words,no_top_documents)
print "NMF"
display_topics(nmf_H,nmf_W,tfidf_feature_names,tweets,no_top_words,no_top_documents)
# Most representative tweet

clusters1=lda_W[:,0]
a=np.array(clusters1)
b=np.array(lda_W[:,1])
c=lda_W[:,0]+lda_W[:,1]
print nmf_W[:,1].shape
clusters2=nmf_W[:,1]

values1=[]
values2=[]
countC1=0
countC2=0
for i, j in lda_W:
    #print i,j
    if(i>0.5):
        countC1+=1
    else:
        countC2+=1
print "Counts "
print countC1,countC2
countC1=0
countC2=0
for i, j in nmf_W:
    print i,j
    if(i>0.03):
        values2.append(0)
        countC1 += 1
    else:
        values2.append(1)
        countC2+=1
print "Counts "
print countC1,countC2
#print values2
# Predict#
print "Silhouette"
print metrics.silhouette_score(tf, lda_W[:,1], metric='euclidean')

print "Sil 2"
print metrics.silhouette_score(tfidf, values2, metric='euclidean')


# save model
filename = 'finalized_model21_lda.sav'
joblib.dump(lda, open(filename, 'w'))
filename = 'finalized_model21_lda.pkl'
joblib.dump(lda, open(filename, 'w'))
# Visualize
MDS()
mds=MDS(n_components=2,dissimilarity='precomputed',random_state=1)
pos=MDS=mds.fit_transform(dist)
xs, ys=pos[:,0],pos[:,1]
print()

cluster_colors={0:'#7570b3',1:'#66a61e',}
cluster_names={0:'cluster 1',1:'cluster 2'}

df=pd.DataFrame(dict(x=xs,y=ys,label=values2))
groups=df.groupby('label')

fig,ax=plt1.subplots(figsize=(5,5))
ax.margins(0.05)
for name, group in groups:
    ax.plot(group.x,group.y,marker='o',linestyle='',ms=12,
            label=cluster_names[name], color=cluster_colors[name],mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis='y',which='both',left='off',top='off',labelleft='off')
ax.legend(numpoints=1)


plt1.show()