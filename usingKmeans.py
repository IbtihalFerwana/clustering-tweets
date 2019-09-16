#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import numpy as np
from sklearn.externals import joblib
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score,silhouette_samples
tweets = []
# reding from a file
#text_file = io.open("clean_tweets_30_11.txt", "r", encoding="UTF-8-sig")
text_file=io.open("combined_tweets_21_12.txt","r",encoding="UTF-8-sig")
lines = text_file.read().split('***')
for ln in lines:
    tweets.append(ln)
# start tf-idf
vect = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
text_fitted = vect.fit(tweets)
Xfitted = vect.fit_transform(tweets)

print 'shape *******'
print Xfitted.shape
dist = 1 - cosine_similarity(Xfitted)
print 'Distance'
print dist

# silhoutter_score

# plt.show()


#print vect.vocabulary_
  #  print v.encode('UTF-8')
idf = vect.idf_

rr = dict(zip(vect.get_feature_names(), idf))
#for i in range (len(rr)):
   #print rr.values().__getitem__(i) , rr.keys().pop(i)

feature_names = np.array(vect.get_feature_names())

# Start Clustering #
n_k = 2
model = KMeans(n_clusters=n_k, init='k-means++', max_iter=118, n_init=1)
model.fit(Xfitted)
clusters = model.labels_.tolist()
print model.labels_
count0=0
count1=0
for i in model.labels_:
    if(i==0):
        count0+=1
    else:
        count1+=1
print "COUNTS ____"
print count0,count1
print "** model Inertia ** "
cls = {'cluster': clusters, 'names': feature_names}

# print top tf-idf words #
print "Top temps per cluster"
cen = model.cluster_centers_.argsort()[:, ::-1]
print cen
terms = vect.get_feature_names()
print
print "K-means"
for i in range(n_k):
    ClusterN = "Cluster %d:" % i
    print ClusterN
    for ind in cen[i, :6]:
        print'%s' % terms[ind]
    # for title in
    #    fname_in='corpusTweetsCluster.txt'
    #    with open(fname_in, 'a') as fin:
    #        writer = csv.writer(fin)
    #        for tweet in tweets:
    #            print tweet
    #            if terms[ind] in tweet:
    #                writer.writerow([ClusterN]+tweet.decode("UTF-8"))
    print

sorted_by_idf = np.argsort(vect.idf_)

# Printing highest and lowest idf and tf-idf
new1 = vect.transform(tweets)
sort_by_tfidf = np.argsort(new1.toarray()).flatten()[::-1]
print '....'
print 'Features with lowest idf:\n{}'.format(feature_names[sorted_by_idf[:3]])
print '\nFeatures with highest idf:\n{}'.format(feature_names[sorted_by_idf[-3:]])

print 'Features with lowest tfidf:\n{}'.format(feature_names[sort_by_tfidf[:3]])
print '\nFeatures with highest tfidf:\n{}'.format(feature_names[sort_by_tfidf[-3:]])

# Visualizing highest tf-idf
print "Silhouette"
print metrics.silhouette_score(Xfitted, clusters, metric='euclidean')
#print metrics.calinski_harabaz_score(Xfitted,clusters)
#Validate

# Visualize


# Predict#

# save model
filename = 'finalized_model21.sav'
joblib.dump(model, open(filename, 'w'))
filename = 'finalized_model21.pkl'
joblib.dump(model, open(filename, 'w'))
