import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
import datetime
pd.set_option('display.max_colwidth', 100)
stopword = nltk.corpus.stopwords.words('english')
nltk.download('stopwords')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv('results.csv', error_bad_lines=False)
print(df)
BtcDF=pd.read_csv('cryptoresults.csv',error_bad_lines=False,header = None) 
df.drop(['Username', 'Followers'], axis=1)

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Tweets'] = df['Tweets'].apply(lambda x: remove_punct(x))

def remove_url(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www.\S+", "", text)
    return text
    
df['Tweets'] = df['Tweets'].apply(lambda x: remove_url(x))
 
def clean_tweet(text):
    if type(text) == np.float:
        return ""
    text = text.lower()
    text = re.sub("'", "", text) 
    text = re.sub("@[A-Za-z0-9_]+","", text)
    text = re.sub("#[A-Za-z0-9_]+","", text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[()!?]', ' ', text)
    text = re.sub('\[.*?\]',' ', text)
    text = re.sub("[^a-z0-9]"," ", text)
    text = text.split()
    text = [w for w in text if not w in stopword]
    text = " ".join(word for word in text)
    return text

df['Tweets'] = df['Tweets'].apply(lambda x: clean_tweet(x))

#print(df.head(10))


analyzer = SentimentIntensityAnalyzer()

scores = []



for i in range(df['Tweets'].shape[0]):
   
    compound = analyzer.polarity_scores(df['Tweets'][i])["compound"]
    pos = analyzer.polarity_scores(df['Tweets'][i])["pos"]
    neu = analyzer.polarity_scores(df['Tweets'][i])["neu"]
    neg = analyzer.polarity_scores(df['Tweets'][i])["neg"]
    
    scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                  })

sentiments_score = pd.DataFrame.from_dict(scores)
df = df.join(sentiments_score)

#print(scores)

print(df.head(3))


#print(df.head(10))
df.to_csv('test5.csv')