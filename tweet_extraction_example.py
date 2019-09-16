import re
import io
import csv
import tweepy
from tweepy import OAuthHandler

# authorization info taken from twitter developer account
access_token="xxxx"
access_token_secret="xxxx"
consumer_key="xxxx"
consumer_secret="xxxxx"


auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True)

# Example of querying tweets talking about education
query=u'#education'
# sepecifying filtering details
search=tweepy.Cursor(api.search, q=query+' -RT',lang='ar').items(10)
count=0
temp=[]
# writing tweets to a file
target=io.open('encoded_tweets_sep2019.txt','w',encoding='utf-8')
for item in search:
    print (item.text)
    print (item.created_at)
    line = re.sub("[^A-Za-z]","",item.text)
    dict={'Tweet created at':unicode(item.created_at),
          'Tweet Text':item.text.encode('utf-8'),
          #'User Location':item.coordinates
          }
    temp.append(dict)
    temp.append(item)
    count+=1
    target.write(line+"\n")
print(count)
print(temp)
