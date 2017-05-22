import os
from TwitterSearch import *
import csv
import time

#define the search function
def SearchTweets(searchkey, round):
    try:
        tso = TwitterSearchOrder()  # create a TwitterSearchOrder object
        tso.set_keywords([searchkey])  # let's define all words we would like to have a look for
        tso.set_language('en')  # we want to see German tweets only
        tso.set_include_entities(False)  # and don't give us all those entity information
		
        # it's about time to create a TwitterSearch object with our secret tokens
        ts = TwitterSearch(
            consumer_key='sToxW29MQSeHNGMHkLJd4upux?',
            consumer_secret='5BbJoy8pRx3WvvI5zZD09YFpHmUaXiPl2AoNCSCPqS37oVKEI1?',
            access_token='798172108010508288-ZKwC04JG1BaT65BhFyi792yA7HadGIK?',
            access_token_secret='nEDYhL7NNzXpaJgwtgJJ9ipJMrLm2IqUwNJJY0Xk6Yc2P?'
        )
        filename = searchkey + '_' +  str(round)
        fieldnames = ['category','user', 'tweet', 'time']
        with open('%s_tweets.csv' % filename, 'w', encoding="utf-8", newline='') as csvWfile:
            writer = csv.DictWriter(csvWfile, fieldnames=fieldnames)
            writer.writeheader()
            # this is where the fun actually starts :)
            for tweet in ts.search_tweets_iterable(tso):
                writer.writerow({'category': searchkey,'user': tweet['user']['screen_name'], 'tweet': tweet['text'], 'time': tweet['created_at']})
    except TwitterSearchException as e:  # take care of all those ugly errors if there are some
           print(e)

#defind the keywords for cotegories
keylist = []
with open('keywords.csv') as csvRfile:
    reader = csv.DictReader(csvRfile)
    for row in reader:
        keylist.append(row['keywords'])
print (keylist)
index = 42
os.chdir('tweets/')
while(index < 700):
     for word in keylist:
        SearchTweets(word, index)
        time.sleep(1200)
     index += 1
