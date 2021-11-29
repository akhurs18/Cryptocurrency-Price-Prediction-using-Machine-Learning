import tweepy
import csv
import pandas as pd
import preprocessor as p


consumer_key = 'nlqDhuBR7PY8Zzx9dWvK28HIZ'
consumer_secret = 'KOQ0w3MyHZEx5lVBO7kpvC6szjhHfCvMKTmcGkJ1c8WCIHju9Z'
access_token='1436013967298727938-o51gaLW95TcbewpROrVR0b8dyt7fER'
access_token_secret='K1fc5X4AndC7lGuBhREfa1hMGRrCmd4gVJhx6DKCwfoHB'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

startSince = '2021-10-04'
endUntil = '2021-11-05'
#Use csv Writer
q="#BTC OR #CARDANO"
csvWriter = csv.writer(open("results.csv", "a",encoding='utf-8-sig',newline=''))
csvWriter.writerow(["Date", "Tweets", "Username", "Followers"])

for tweet in tweepy.Cursor(api.search,q, count=100,
                           lang="en",
                           since= startSince,until= endUntil).items(120000):
                           
    if tweet.user.followers_count > 100:                      
        print (tweet.user.name,tweet.user.followers_count,tweet.created_at, tweet.text)
        csvWriter.writerow([tweet.created_at, tweet.text,tweet.user.name,tweet.user.followers_count])


